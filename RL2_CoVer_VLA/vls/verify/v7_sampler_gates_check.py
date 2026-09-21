"""V7: with guidance OFF, GuidedSampler.sample() is plain flow matching.

V5 checks the steering *math* against a live SIMPLER scene. This checks the
sampler's *control flow* with a stub model, so it needs no env and runs anywhere.

Why it matters: upstream VLS bypasses the guided sampler for guidance-OFF chunks
(``predict_action_chunk``, pi05_steer.py:124-125). This port reuses the guided
loop for them (so pi0's velocity stays observable for the overlay), which is only
faithful if EVERY steering term is gated on ``use_guidance``. A term that is not
gated (the RBF diversity term was one) keeps perturbing the executed particle on
the hand-off chunks, and is invisible in the video because the overlay only draws
the keypoint-gradient term.

  V7a  guidance OFF: no diversity, no keypoint gradient, no FKD, and the output is
       bit-identical to a run with diversity disabled (= plain flow matching).
  V7b  guidance ON: diversity on the steps with time > start_time, keypoint
       gradient on the rest (the two-phase schedule is unchanged).

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

B, CH, MAXD, D, HORIZON, NUM_STEPS = 5, 50, 32, 7, 4, 10
START_TIME = 0.8
FKD_CFG = {"potential_type": "max", "lmbda": 10.0, "adaptive_resampling": True,
           "resample_frequency": 5}


# ---- stubs: only the surface GuidedSampler.sample() touches ------------------
class _Model:
    config = types.SimpleNamespace(num_steps=NUM_STEPS, chunk_size=CH, max_action_dim=MAXD)
    _target = torch.randn(1, CH, MAXD, generator=torch.Generator().manual_seed(1))

    def sample_noise(self, shape, device, noise_std=1.0):
        return torch.randn(shape) * noise_std

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        return x_t - self._target, None            # deterministic in x_t


class _Policy:
    config = types.SimpleNamespace(n_action_steps=HORIZON,
                                   action_feature=types.SimpleNamespace(shape=(D,)))
    model = _Model()

    def forward_pass_vlm(self, obs, noise_std=1.0):
        return {}, torch.zeros(B, 8), torch.ones(B, 4, dtype=torch.bool)

    def unnormalize_outputs(self, d):
        return d


class _Adapter:
    _act_scale = torch.tensor([0.01, 0.01, 0.01])
    _base_rot = np.eye(3, dtype=np.float32)

    def delta_actions_to_ee_trajectory(self, a):
        return torch.cat([torch.zeros(1, 3), torch.cumsum(a[:, :3] * self._act_scale, 0)], 0)


def _guidance(kp, traj):
    return -1000.0 * ((traj - kp[0]) ** 2).sum(dim=-1).mean()


KP = np.zeros((6, 3), dtype=np.float32)
OBS = {"observation.images.top": torch.zeros(1, 3, 4, 4),
       "observation.state": torch.zeros(1, 8)}

# ---- spies ---------------------------------------------------------------------
calls = {"div": 0, "kp": 0, "fkd": 0}
_o_div, _o_kp, _o_fkd = P.compute_diversity_gradient, P.compute_keypoint_gradient, P.FKD


def _div(*a, **k):
    calls["div"] += 1
    return _o_div(*a, **k)


def _kp(*a, **k):
    calls["kp"] += 1
    return _o_kp(*a, **k)


def _fkd(**kw):
    calls["fkd"] += 1
    return _o_fkd(**kw)


P.compute_diversity_gradient, P.compute_keypoint_gradient, P.FKD = _div, _kp, _fkd


def run(guidance_fn, use_diversity=True, use_fkd=True, seed=0):
    calls.update(div=0, kp=0, fkd=0)
    torch.manual_seed(seed)
    sampler = P.GuidedSampler(_Policy(), _Adapter(), None)
    out = sampler.sample(
        processed_obs=OBS, image_key="img", task_list=["t"] * B, batch_size=B, keypoints=KP,
        guidance_fn=guidance_fn, action_noise_std=1.0, guide_scale=80.0, diversity_scale=20.0,
        start_ratio=None, sigmoid_k=25.0, sigmoid_x0=0.75, use_diversity=use_diversity,
        use_fkd=use_fkd, fkd_config=FKD_CFG, verbose=False,
    )
    return out, sampler


ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{label:<66} {'PASS' if cond else 'FAIL'}  {detail}")


# ---- V7a: OFF == plain flow matching --------------------------------------------
off, _ = run(None, use_diversity=True, use_fkd=True)
check("V7a OFF: no diversity / keypoint-gradient calls, FKD not built",
      calls["div"] == 0 and calls["kp"] == 0 and calls["fkd"] == 0,
      f"(div={calls['div']} kp={calls['kp']} fkd={calls['fkd']})")
plain, _ = run(None, use_diversity=False, use_fkd=False)
check("V7a OFF output bit-identical to diversity-disabled run", torch.equal(off, plain))

# ---- V7b: ON schedule unchanged --------------------------------------------------
n_div_steps = sum(1 for i in range(NUM_STEPS) if 1.0 - i / NUM_STEPS > START_TIME + 1e-6)
_, sampler = run([_guidance], use_fkd=False)
check("V7b ON: diversity on the early steps, keypoint gradient on the rest",
      calls["div"] == n_div_steps and calls["kp"] == NUM_STEPS - n_div_steps,
      f"(div={calls['div']} kp={calls['kp']})")
check("V7b ON overlay records pi0 + vls terms", sorted(sampler.get_last_terms()) == ["pi0", "vls"])

print("\nV7:", "ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
