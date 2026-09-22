"""V7: control flow of GuidedSampler.sample() -- steering gates and FKD wiring.

V5 checks the steering *math* against a live SIMPLER scene. This checks the
sampler's *control flow* with a stub model, so it needs no env and runs anywhere.
It drives the real ``sample()`` / ``_init_fkd``, which is the point: V5d exercises
the FKD class directly and never sees the sampler's time grid, so it cannot catch
a mis-wired grid.

  V7a  guidance OFF is plain flow matching: no diversity, no keypoint gradient,
       no FKD, and the output is bit-identical to a run with diversity disabled.
  V7b  guidance ON: diversity runs on the steps with time > start_time, keypoint
       gradient on the rest.
  V7c  FKD scores rewards on the expected steps, reaches its terminal step, and
       leaves the best (max population_rs) particle at index 0. Regression test
       for (1) upstream pi05's ``linspace(...).long()`` grid, which collapses to
       [1,0,0,...] so nothing ever resamples, and (2) an (n+1)-entry grid whose
       terminal index is one past the last step the loop takes.
  V7d  FKD.last_indices is the identity on calls that do not resample; a stale
       permutation would be re-applied to per-particle state on every such call.
  V7e  per-particle state (KV cache, `state`) stays aligned with the resampled
       particles through every mid-loop resample, not just the terminal one.

Usage:
    python vls/verify/v7_sampler_gates_check.py
"""
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # RL2_CoVer_VLA/

import vls.core.pi0_steer as P  # noqa: E402
from vls.core.fkd_class import FKD  # noqa: E402

B, CH, MAXD, D, HORIZON, NUM_STEPS = 5, 50, 32, 7, 4, 10
START_TIME = 0.8
FKD_CFG = {"potential_type": "max", "lmbda": 10.0, "adaptive_resampling": True,
           "resample_frequency": 5}


# V7e tags every particle with its id in x_t's last (unused) channel, its KV cache
# and `state`, then checks they stay aligned through every FKD resample.
TRACK = {"on": False, "misaligned": 0, "denoise_calls": 0}


# ---- stubs: only the surface GuidedSampler.sample() touches ------------------
class _Model:
    config = types.SimpleNamespace(num_steps=NUM_STEPS, chunk_size=CH, max_action_dim=MAXD)
    _target = torch.randn(1, CH, MAXD, generator=torch.Generator().manual_seed(1))

    def sample_noise(self, shape, device, noise_std=1.0):
        n = torch.randn(shape) * noise_std
        if TRACK["on"]:
            n[:, 0, -1] = torch.arange(shape[0], dtype=n.dtype)
        return n

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        v = x_t - self._target                     # deterministic in x_t
        if TRACK["on"]:
            v[:, :, -1] = 0.0                      # keep the id channel constant
            kv_ids = past_key_values["l0"]["key_states"][:, 0, 0]
            TRACK["denoise_calls"] += 1
            TRACK["misaligned"] += int(not (torch.equal(kv_ids, x_t[:, 0, -1])
                                            and torch.equal(state[:, 0], x_t[:, 0, -1])))
        return v, None


class _Policy:
    config = types.SimpleNamespace(n_action_steps=HORIZON,
                                   action_feature=types.SimpleNamespace(shape=(D,)))
    model = _Model()

    def forward_pass_vlm(self, obs, noise_std=1.0):
        state = torch.zeros(B, 8)
        if not TRACK["on"]:
            return {}, state, torch.ones(B, 4, dtype=torch.bool)
        ids = torch.arange(B, dtype=torch.float32)
        state[:, 0] = ids
        kv = {"l0": {"key_states": ids.view(B, 1, 1).clone(), "value_states": ids.view(B, 1, 1).clone()}}
        return kv, state, torch.ones(B, 4, dtype=torch.bool)

    def unnormalize_outputs(self, d):
        return d


class _Adapter:
    _act_scale = torch.tensor([0.01, 0.01, 0.01])
    _base_rot = np.eye(3, dtype=np.float32)

    def delta_actions_to_ee_trajectory(self, a):
        return torch.cat([torch.zeros(1, 3), torch.cumsum(a[:, :3] * self._act_scale, 0)], 0)


def _guidance(kp, traj):                            # reward O(1), spread across particles
    return -1000.0 * ((traj - kp[0]) ** 2).sum(dim=-1).mean()


KP = np.zeros((6, 3), dtype=np.float32)
OBS = {"observation.images.top": torch.zeros(1, 3, 4, 4),
       "observation.state": torch.zeros(1, 8)}

# ---- spies ---------------------------------------------------------------------
calls = {"div": 0, "kp": 0, "scored": [], "fkd": None}
_o_div, _o_kp, _o_fkd, _o_rew = (P.compute_diversity_gradient, P.compute_keypoint_gradient,
                                 P.FKD, FKD._compute_reward)


def _div(*a, **k):
    calls["div"] += 1
    return _o_div(*a, **k)


def _kp(*a, **k):
    calls["kp"] += 1
    return _o_kp(*a, **k)


def _fkd(**kw):
    calls["fkd"] = _o_fkd(**kw)
    return calls["fkd"]


def _rew(self, x0):
    calls["scored"].append(self._last_idx_sampled)   # set before the reward is computed
    return _o_rew(self, x0)


P.compute_diversity_gradient, P.compute_keypoint_gradient, P.FKD = _div, _kp, _fkd
FKD._compute_reward = _rew


def run(guidance_fn, use_diversity=True, use_fkd=True, seed=0, fkd_cfg=FKD_CFG):
    calls.update(div=0, kp=0, scored=[], fkd=None)
    torch.manual_seed(seed)
    sampler = P.GuidedSampler(_Policy(), _Adapter(), None)
    out = sampler.sample(
        processed_obs=OBS, image_key="img", task_list=["t"] * B, batch_size=B, keypoints=KP,
        guidance_fn=guidance_fn, action_noise_std=1.0, guide_scale=80.0, diversity_scale=20.0,
        start_ratio=None, sigmoid_k=25.0, sigmoid_x0=0.75, use_diversity=use_diversity,
        use_fkd=use_fkd, fkd_config=fkd_cfg, verbose=False,
    )
    return out, sampler


ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{label:<64} {'PASS' if cond else 'FAIL'}  {detail}")


# ---- V7a: OFF == plain flow matching --------------------------------------------
off, sampler = run(None, use_diversity=True, use_fkd=True)
check("V7a OFF: no diversity / keypoint-grad calls, FKD not built",
      calls["div"] == 0 and calls["kp"] == 0 and calls["fkd"] is None,
      f"(div={calls['div']} kp={calls['kp']} fkd_built={calls['fkd'] is not None})")
plain, _ = run(None, use_diversity=False, use_fkd=False)
check("V7a OFF output bit-identical to diversity-disabled run", torch.equal(off, plain))
check("V7a OFF overlay records only the pi0 term", sorted(sampler.get_last_terms()) == ["pi0"])

# ---- V7b: ON -> diversity on time > start_time, keypoint gradient after ----------
n_div_steps = sum(1 for i in range(NUM_STEPS) if 1.0 - i / NUM_STEPS > START_TIME + 1e-6)
_, sampler = run([_guidance], use_fkd=False)
check("V7b ON: diversity on the early steps, keypoint gradient on the rest",
      calls["div"] == n_div_steps and calls["kp"] == NUM_STEPS - n_div_steps,
      f"(div={calls['div']} kp={calls['kp']})")
check("V7b ON overlay records pi0 + vls terms", sorted(sampler.get_last_terms()) == ["pi0", "vls"])

# ---- V7c: FKD wiring through the sampler's real grid -----------------------------
start_idx = NUM_STEPS - int(round(START_TIME * NUM_STEPS))
expected = sorted(set(range(start_idx, NUM_STEPS, FKD_CFG["resample_frequency"])) | {NUM_STEPS - 1})
seeds, sorted_ok, wiring_ok, best_final = range(10), True, True, 0
for s in seeds:
    out, _ = run([_guidance], use_fkd=True, seed=s)
    fkd = calls["fkd"]
    rs = fkd.population_rs
    if calls["scored"] != expected or not fkd.reached_terminal:
        wiring_ok = False
        print(f"  seed {s}: scored={calls['scored']} terminal={fkd.reached_terminal}")
    sorted_ok &= bool(torch.all(rs[:-1] >= rs[1:])) and bool(rs[0] == rs.max())
    fin = torch.stack([_guidance(torch.as_tensor(KP), _Adapter().delta_actions_to_ee_trajectory(
        out[i, :HORIZON, :D])[1:HORIZON][None]) for i in range(B)])
    best_final += int(fin[0] >= fin.max() - 1e-6)
check(f"V7c FKD scores rewards exactly on steps {expected}, terminal reached",
      wiring_ok, f"({len(seeds)} seeds)")
check("V7c terminal sort: population_rs descending, particle 0 is max", sorted_ok)
print(f"     (info) particle 0 also has the best FINAL reward in {best_final}/{len(seeds)} seeds "
      f"-- the 'max' potential ranks by best-so-far reward, so this need not be 100%")

# ---- V7d: last_indices is identity on non-resampling calls -------------------------
torch.manual_seed(0)
rewards = torch.linspace(-1.0, 0.0, B)
fk = FKD(potential_type="max", lmbda=10.0, num_particles=B, adaptive_resampling=False,
         resample_frequency=FKD_CFG["resample_frequency"], resampling_t_start=start_idx,
         resampling_t_end=NUM_STEPS - 1, timesteps=torch.arange(NUM_STEPS),
         reward_fn=lambda x0: rewards.clone(), reward_min_value=float("-inf"), device="cpu")
x, ident, moved_at, stale = torch.randn(B, 4, 7), torch.arange(B), [], []
for step in range(NUM_STEPS):
    x, scored = fk.resample(sampling_idx=step, latents=x, x0_preds=x)
    if scored is None and not torch.equal(fk.last_indices, ident):
        stale.append(step)
    if not torch.equal(fk.last_indices, ident):
        moved_at.append(step)
check("V7d non-resampling calls leave last_indices as identity", not stale,
      f"(stale at {stale}; moved at {moved_at})")
check("V7d resampling calls do move particles", len(moved_at) >= 1)

# ---- V7e: per-particle state follows the resampled particles ----------------------
# adaptive_resampling=False forces a real resample at every interval step, so the
# mid-loop reindex of KV cache / state is exercised (not just the terminal one).
TRACK.update(on=True, misaligned=0, denoise_calls=0)
reindexed = {"n": 0}
_o_reindex = P._reindex_particles
P._reindex_particles = lambda *a, **k: (reindexed.__setitem__("n", reindexed["n"] + 1), _o_reindex(*a, **k))[1]
for s in range(10):
    run([_guidance], use_fkd=True, seed=s, fkd_cfg={**FKD_CFG, "adaptive_resampling": False})
P._reindex_particles = _o_reindex
TRACK["on"] = False
check("V7e KV cache / state stay aligned with resampled particles",
      TRACK["misaligned"] == 0 and reindexed["n"] >= 10,
      f"({TRACK['denoise_calls']} denoise steps, {reindexed['n']} reindexes, "
      f"{TRACK['misaligned']} misaligned)")

print("\nV7:", "ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
