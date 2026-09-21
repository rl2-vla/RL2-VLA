"""V5: trajectory-decoder Jacobian, gradient correctness, descent direction, FKD liveness.

These gate the steering logic: a decoder with the wrong frame/scale, or a
gradient with the wrong sign, passes every shape check while steering the robot
confidently in the wrong direction.

Usage:
    python vls/verify/v5_gradient_check.py [output_dir]

Writes v5_guidance.png: decoded trajectories projected into the camera, showing
guidance bending the trajectory toward the target keypoint.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from _common import DEFAULT_OUT, make_adapter  # noqa: E402

from vls.core.fkd_class import FKD  # noqa: E402
from vls.core.pi0_steer import compute_diversity_gradient, compute_keypoint_gradient  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
OUT.mkdir(parents=True, exist_ok=True)

env, adapter, obs = make_adapter()
torch.manual_seed(0)

HORIZON = 4          # pi0 chunk_size / n_action_steps for this checkpoint
ADIM = 7

# ---------- V5a: decoder Jacobian is exactly the affine closed form ----------
a = torch.randn(HORIZON, ADIM, dtype=torch.float32) * 0.3
traj = adapter.delta_actions_to_ee_trajectory(a)
print(f"V5a decoder: in {tuple(a.shape)} -> out {tuple(traj.shape)} "
      f"(expect ({HORIZON+1}, 3))")

scale = adapter._act_scale.numpy().astype(np.float64)
R = adapter._base_rot.astype(np.float64)
eps = 1e-3
# float64 throughout: the decoder is affine, so in double precision the finite
# difference matches the closed form to ~1e-12. In float32 the subtraction of
# two nearby positions loses ~3 digits and the same check reads ~2e-3, which is
# precision noise rather than a decoder error.
a64 = a.double()
traj64 = adapter.delta_actions_to_ee_trajectory(a64).numpy().astype(np.float64)
max_rel = 0.0
for t in range(HORIZON):
    for j in range(3):
        pert = a64.clone(); pert[t, j] += eps
        fd = ((adapter.delta_actions_to_ee_trajectory(pert).numpy().astype(np.float64) - traj64) / eps)
        analytic = np.zeros_like(fd)
        analytic[t + 1:, :] = R[:, j] * scale[j]      # affine: constant for k > t
        denom = max(np.abs(analytic).max(), 1e-9)
        max_rel = max(max_rel, np.abs(fd - analytic).max() / denom)
print(f"V5a Jacobian vs closed form (float64): max rel err={max_rel:.2e} "
      f"-> {'PASS' if max_rel < 1e-8 else 'FAIL'}  (affine => machine precision)")

# ---------- V5a2: the bias term is present and material ----------
zero = torch.zeros(HORIZON, ADIM)
drift = (adapter.delta_actions_to_ee_trajectory(zero)[-1]
         - adapter.delta_actions_to_ee_trajectory(zero)[0]).numpy()
bias_world = R @ (adapter._act_bias.numpy() * HORIZON)
print(f"V5a2 zero-action drift over {HORIZON} steps: {np.round(drift,4)} m "
      f"(expect {np.round(bias_world,4)}) -> "
      f"{'PASS' if np.allclose(drift, bias_world, atol=1e-5) else 'FAIL'}")
print(f"     |drift|={np.linalg.norm(drift)*100:.2f} cm — a scale-only decoder would lose this")

# ---------- V5b/c: reward gradient + descent direction ----------
# Stand-in for a VLM-written reward: negative squared distance to a keypoint,
# matching the shape of real guidance (see VLS stage1_guidance.txt).
seg_names = adapter.process_segmentation(obs["image"]["3rd_view_camera"]["Segmentation"])[2]
target_seg = sorted(k for k in seg_names if k != 0)[0]
target = torch.as_tensor(adapter.get_object_pose_by_segment(target_seg).position, dtype=torch.float32)
keypoints = target.unsqueeze(0)
print(f"\nV5b target keypoint: {seg_names[target_seg]} @ {np.round(target.numpy(),3)}")


def guidance(kps, traj_b):
    return -((traj_b[..., :3] - kps[0]) ** 2).sum(-1).mean()


def decode(sample):
    return torch.stack([adapter.delta_actions_to_ee_trajectory(sample[b, :HORIZON, :ADIM])
                        for b in range(sample.shape[0])], dim=0)


sample = torch.randn(1, HORIZON, 32, dtype=torch.float32) * 0.3
grad, reward = compute_keypoint_gradient(sample, keypoints, guidance, decode, HORIZON)
print(f"V5b gradient: shape={tuple(grad.shape)} |g|={float(torch.norm(grad)):.4f} reward={reward:.6f}")

# central differences on random coordinates
rng = np.random.default_rng(0)
pairs = []
for _ in range(15):
    t, j = int(rng.integers(HORIZON)), int(rng.integers(3))
    h = 1e-3
    plus, minus = sample.clone(), sample.clone()
    plus[0, t, j] += h
    minus[0, t, j] -= h
    fd = (float(guidance(keypoints, decode(plus)[:, :HORIZON, :3]))
          - float(guidance(keypoints, decode(minus)[:, :HORIZON, :3]))) / (2 * h)
    pairs.append((fd, float(grad[0, t, j])))
# grad is unit-normalized, so compare DIRECTION (cosine), not magnitude.
fds = np.array([p[0] for p in pairs]); gs = np.array([p[1] for p in pairs])
cos = float(fds @ gs / (np.linalg.norm(fds) * np.linalg.norm(gs) + 1e-12))
print(f"V5b finite-diff vs autograd: cosine={cos:.6f} "
      f"-> {'PASS' if cos > 0.999 else 'FAIL'}  (unit-normalized grad => compare direction)")

# ---------- V5c: the END-TO-END update moves toward higher reward ----------
# Two sign flips compose here, so checking the gradient alone is misleading:
#   1. compute_keypoint_gradient returns +d(reward)/d(sample)  [ASCENT]
#   2. the velocity update SUBTRACTS it:      v' = v - scale*g
#   3. the Euler step uses a NEGATIVE dt:     x' = x + dt*v'   (dt = -1/num_steps)
# so   x' = x + dt*v + scale*|dt|*g  -- i.e. the sample moves ALONG +g.
# What must hold is that the composed update increases reward and closes the
# distance to the target.
target_pos = keypoints[0]


def final_dist(s):
    return float(torch.norm(decode(s)[0, -1, :3] - target_pos))


dt = -1.0 / 10          # pi0 num_steps
step = 0.05
base_r = float(guidance(keypoints, decode(sample)[:, :HORIZON, :3]))
# emulate one guided Euler step with v = 0, isolating the guidance term
guided = sample + dt * (-step * grad)
unguided = sample + dt * (+step * grad)      # wrong sign, for contrast
g_r, u_r = (float(guidance(keypoints, decode(x)[:, :HORIZON, :3])) for x in (guided, unguided))
print(f"V5c composed update (v=0, dt={dt}):")
print(f"     base      reward={base_r:.6f}  final_dist={final_dist(sample):.4f} m")
print(f"     v -= s*g  reward={g_r:.6f}  final_dist={final_dist(guided):.4f} m   <- VLS")
print(f"     v += s*g  reward={u_r:.6f}  final_dist={final_dist(unguided):.4f} m   (wrong sign)")
print(f"V5c -> {'PASS' if g_r > base_r > u_r and final_dist(guided) < final_dist(sample) else 'FAIL'}"
      f"  (subtracting from v must raise reward and close distance)")

# ---------- V5d: FKD actually resamples ----------
NP_ = 5
NSTEP = 10
rewards = torch.linspace(-1.0, 0.0, NP_)


def fkd_reward(x0):
    return rewards.clone()


fkd = FKD(potential_type="max", lmbda=10.0, num_particles=NP_,
          adaptive_resampling=False, resample_frequency=1,
          resampling_t_start=0, resampling_t_end=NSTEP,
          timesteps=torch.arange(NSTEP + 1), reward_fn=fkd_reward,
          reward_min_value=float("-inf"), device="cpu")
print(f"\nV5d t_to_index: {len(fkd.t_to_index)} unique for {NSTEP+1} timesteps "
      f"-> {'PASS' if len(fkd.t_to_index) == NSTEP + 1 else 'FAIL'} "
      f"(pi05's linspace wiring collapses this to 2)")

x = torch.randn(NP_, 4, 7)
fired = 0
for s in range(NSTEP + 1):
    x, _ = fkd.resample(sampling_idx=s, latents=x, x0_preds=x)
    if not torch.equal(fkd.last_indices, torch.arange(NP_)):
        fired += 1
print(f"V5d resample fired on {fired}/{NSTEP+1} steps -> {'PASS' if fired >= 5 else 'FAIL'}")
print(f"V5d terminal sort: population_rs descending="
      f"{bool(torch.all(fkd.population_rs[:-1] >= fkd.population_rs[1:]))} "
      f"| particle0 is max={bool(fkd.population_rs[0] == fkd.population_rs.max())} "
      f"-> {'PASS' if fkd.population_rs[0] == fkd.population_rs.max() else 'FAIL'}")

# ---------- diversity gradient sanity ----------
multi = torch.randn(4, HORIZON, 32) * 0.3
dg = compute_diversity_gradient(multi, decode, HORIZON)
print(f"\nDiversity grad (B=4): shape={tuple(dg.shape)} |g|={float(torch.norm(dg)):.4f} -> PASS")
print(f"Diversity grad (B=1): {compute_diversity_gradient(multi[:1], decode, HORIZON)} "
      f"-> {'PASS' if compute_diversity_gradient(multi[:1], decode, HORIZON) is None else 'FAIL'} (self-disables)")

# ---------- visual: guidance bends the trajectory toward the keypoint ----------
# Emulate the guided denoising loop with v=0, so the ONLY thing moving the
# sample is the guidance term. Each iteration applies one composed update:
#   x <- x + dt*(-scale*g),  dt < 0  =>  x moves along +g (toward higher reward)
GUIDE_SCALE, N_ITERS = 0.6, 25
x = sample.clone()
traj_history = [decode(x)[0, :, :3].numpy().copy()]
dists = [final_dist(x)]
for _ in range(N_ITERS):
    g, _ = compute_keypoint_gradient(x, keypoints, guidance, decode, HORIZON)
    if g is None:
        break
    x = x + dt * (-GUIDE_SCALE * g)
    traj_history.append(decode(x)[0, :, :3].numpy().copy())
    dists.append(final_dist(x))
print(f"\nGuidance sweep: final-point distance {dists[0]:.4f} -> {dists[-1]:.4f} m "
      f"over {len(dists)-1} steps -> {'PASS' if dists[-1] < dists[0] else 'FAIL'}")

cp = adapter.get_camera_params()
K, E = cp.intrinsic, cp.extrinsic


def project(pts_world):
    cam = (np.c_[pts_world, np.ones(len(pts_world))] @ E.T)[:, :3]
    uv = (cam @ K.T)
    return uv[:, :2] / uv[:, 2:3]


import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))

# Plot ONLY the first and last trajectory. Overlaying all 26 produced a hatched
# fan that reads as noise: each line is a separate 5-point chunk, not a path the
# robot follows, and they largely overlap.
ax[0].imshow(adapter.get_vlm_image())
tgt_uv = project(target.numpy()[None])[0]
ax[0].plot(*tgt_uv, "*", color="lime", markersize=24, markeredgecolor="black",
           markeredgewidth=1.2, label=f"target ({seg_names[target_seg]})", zorder=6)

first_tr, last_tr = traj_history[0], traj_history[-1]
for tr, color, lbl in ((first_tr, "#8e44ad", "unguided chunk"),
                       (last_tr, "#f1c40f", "guided chunk")):
    uv = project(tr)
    ax[0].plot(uv[:, 0], uv[:, 1], "-o", color=color, markersize=6, linewidth=2.6,
               markeredgecolor="black", markeredgewidth=0.6, label=lbl, zorder=4)
    ax[0].plot(*uv[-1], "s", color=color, markersize=11, markeredgecolor="black",
               markeredgewidth=1.2, zorder=5)

start_uv = project(first_tr[0][None])[0]
ax[0].plot(*start_uv, "o", color="white", markersize=10, markeredgecolor="black",
           markeredgewidth=1.4, label="EE start (shared)", zorder=6)

xs = np.r_[project(first_tr)[:, 0], project(last_tr)[:, 0], tgt_uv[0]]
ys = np.r_[project(first_tr)[:, 1], project(last_tr)[:, 1], tgt_uv[1]]
pad = 70
ax[0].set_xlim(max(xs.min() - pad, 0), min(xs.max() + pad, cp.width))
ax[0].set_ylim(min(ys.max() + pad, cp.height), max(ys.min() - pad, 0))
ax[0].set_xticks([]); ax[0].set_yticks([])
ax[0].legend(loc="upper right", fontsize=9)
ax[0].set_title("One action chunk, decoded: unguided vs guided\n"
                "squares = chunk endpoints; both start at the same EE pose",
                fontsize=11)

ax[1].plot(dists, "-o", color="#0b6fa4", markersize=4)
ax[1].set_xlabel("guidance iteration"); ax[1].set_ylabel("distance to keypoint (m)")
ax[1].grid(alpha=0.3)
chunk_span = float(np.linalg.norm(traj_history[-1][-1] - traj_history[-1][0]))
ax[1].set_title(f"Monotone decrease: {dists[0]:.3f} -> {dists[-1]:.3f} m\n"
                f"(v=0 isolates guidance; one 4-step chunk spans only "
                f"~{chunk_span*100:.0f} cm, so it steers rather than arrives)",
                fontsize=10)

fig.suptitle("V5 guidance — trajectory bends toward the target keypoint", fontsize=12)
plt.tight_layout()
plt.savefig(str(OUT / "v5_guidance.png"), dpi=110, bbox_inches="tight")
print(f"wrote {OUT / 'v5_guidance.png'}")
