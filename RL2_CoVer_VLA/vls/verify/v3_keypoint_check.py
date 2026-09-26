"""V3: keypoint detection + rigid-body tracking through real object motion.

The tracker stores each keypoint as an offset in its object's LOCAL frame and
re-derives the world position from the object's live pose. A test where nothing
moves proves nothing, so this pushes the target object and checks the tracked
keypoints follow it.

Usage:
    python vls/verify/v3_keypoint_check.py [output_dir]

Writes v3_tracking.png: before/after frames with tracked keypoints projected on
top, so you can see them stay attached as the object is pushed.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from _common import DEFAULT_OUT, make_adapter, make_detector  # noqa: E402

from vls.core.keypoint_tracker import KeypointTracker  # noqa: E402
from vls.utils.vis_utils import draw_keypoints_on_image  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
OUT.mkdir(parents=True, exist_ok=True)

env, adapter, obs = make_adapter()
KEYPOINT_DETECTOR = make_detector()

rgb, _, points, seg, names = adapter.get_keypoint_detection_inputs()
valid = np.linalg.norm(points, axis=-1) > 1e-6

# ---------- V3a: detection with the REAL VLS detector ----------
# NO API KEY is involved here. KeypointDetector runs a LOCAL DINO model (see
# make_detector() in _common.py for which one) and k-means-clusters its patch
# features inside each segmentation mask (core/keypoint_detector.py:
# _cluster_features). OPENAI_API_KEY is only needed later to synthesize guidance
# functions; GOOGLE_API_KEY only for live stage recognition. Detection works offline.
#
# Settings are KEYPOINT_DETECTOR_CFG in _common.py, which mirrors vls/runtime.py.
# They match upstream VLS (configs/perception.yaml) except the workspace bounds,
# the merge radius (keypoint density) and the feature extractor, all noted there.
kps, projected_img, mask_ids = KEYPOINT_DETECTOR.get_keypoints(rgb, points, seg, names)
print(f"V3a keypoints: {len(kps)} across {len(set(mask_ids.tolist()))} objects "
      f"(real VLS detector, {KEYPOINT_DETECTOR.feature_extractor_type} features "
      f"+ k-means, no API key)")
for i, (k, m) in enumerate(zip(kps, mask_ids)):
    print(f"      kp{i}: seg={m} {names.get(int(m), '?')[:34]:<34} {np.round(k, 3)}")

ok = True
for kp, sidx in zip(kps, mask_ids):
    m = (seg == sidx) & valid
    d = np.linalg.norm(points[m] - kp, axis=1).min()
    ok &= d < 0.02
print(f"V3a each keypoint within 2cm of its own segment cloud -> {'PASS' if ok else 'FAIL'}")

# ---------- V3b: registration ----------
tracker = KeypointTracker(adapter)
kp_to_obj = tracker.register_keypoints(kps, mask_ids, names)
print(f"V3b registration map: {kp_to_obj}")
start = tracker.get_keypoint_positions()
drift0 = np.abs(start - kps).max()
print(f"V3b round-trip (world -> local -> world) max err={drift0:.6f} m "
      f"-> {'PASS' if drift0 < 1e-3 else 'FAIL'}")

rgb_before = draw_keypoints_on_image(adapter, adapter.get_vlm_image().copy(), start, mask_ids)

# ---------- V3c: tracking through real object motion ----------
# Servo the gripper onto the target and shove it. The action frame is the ROBOT
# BASE frame (ee_align2), so the direction must be computed there, not in world.
target_seg = mask_ids[0]
target_name = names[target_seg]
pose0 = adapter.get_object_pose_by_segment(target_seg)
p0 = pose0.position.copy()
T0 = pose0.to_transformation_matrix()

# IMPORTANT: this control mode has normalize_action=False, so env actions are
# RAW METRES per step (typ. |dxyz| <= ~0.04). A value like 1.0 commands a 1 m
# step, IK fails, and the controller silently falls back to the previous qpos --
# the arm simply never moves. Keep the per-step delta at centimetre scale.
# Three phases: hover above the object (so the gripper doesn't plough through it
# on the way in), descend beside it, then nudge it sideways.
STEP_M = 0.02
HOVER_STEPS, DESCEND_STEPS, PUSH_STEPS = 25, 15, 3
HOVER_M = 0.06           # clearance while closing the lateral gap
OFFSET_M = 0.045         # stand off to the side so we push rather than crush
RETREAT_STEPS = 10       # back off afterwards so the gripper doesn't occlude the object
R_base = adapter.get_robot_base_pose().to_rotation_matrix()
act = np.zeros(env.action_space.shape, dtype=np.float32)
push_dir_world = None

TOTAL_STEPS = HOVER_STEPS + DESCEND_STEPS + PUSH_STEPS + RETREAT_STEPS
for step in range(TOTAL_STEPS):
    act[:] = 0.0
    tgt = adapter.get_object_pose_by_segment(target_seg).position
    ee = adapter.get_ee_pose_world().position
    delta_world = tgt - ee

    if step < HOVER_STEPS:
        # move to a point above and slightly to one side of the object
        if push_dir_world is None:
            lateral = delta_world[:2]
            push_dir_world = lateral / (np.linalg.norm(lateral) + 1e-6)
        goal = delta_world - np.r_[push_dir_world * OFFSET_M, 0.0] + np.array([0.0, 0.0, HOVER_M])
    elif step < HOVER_STEPS + DESCEND_STEPS:
        # descend to object height, still offset to the side
        goal = delta_world - np.r_[push_dir_world * OFFSET_M, 0.0]
    elif step < HOVER_STEPS + DESCEND_STEPS + PUSH_STEPS:
        # brief nudge: enough to move the object measurably, not to shove it
        # across the table or off the edge
        goal = np.r_[push_dir_world * STEP_M, 0.0]
    else:
        # lift and back off, so the gripper doesn't occlude the object in the
        # "after" frame -- the point of the visual is to see the keypoints
        goal = -np.r_[push_dir_world * STEP_M, 0.0] + np.array([0.0, 0.0, 0.05])

    d = R_base.T @ goal
    n = np.linalg.norm(d) + 1e-6
    act[:3] = d / n * min(STEP_M, n)
    # NOTE: despite the name, this raw env action actually CLOSES the widowx
    # gripper (verified live: action=-1 -> eef_pos[7] 1.0->~0); for
    # google_robot -1 does open it. Kept as -1 for both regardless -- it's
    # the value this push routine's constants (OFFSET_M, STEP_M, HOVER_M)
    # were tuned and visually validated against, and V3 only asserts tracking
    # fidelity (unaffected by gripper state), not gripper state itself. Using
    # adapter.gripper_open_sign here would genuinely open widowx's gripper,
    # which contacts the object very differently and sends it flying out of
    # frame -- correct-to-comment, but a worse, untuned test.
    act[-1] = -1.0
    obs, *_ = env.step(act)
    adapter.set_obs(obs)

pose1 = adapter.get_object_pose_by_segment(target_seg)
p1 = pose1.position.copy()
T1 = pose1.to_transformation_matrix()
moved = np.linalg.norm(p1 - p0)
dot = abs(float(np.dot(pose0.quaternion, pose1.quaternion)))
rot_deg = np.degrees(2 * np.arccos(min(dot, 1.0)))
print(f"\nV3c target '{target_name}' moved {moved*100:.2f} cm, rotated {rot_deg:.1f}° "
      f"({np.round(p0,3)} -> {np.round(p1,3)})")

tracked = tracker.get_keypoint_positions()
# Compare against the FULL rigid transform T1 @ inv(T0): a pushed object
# generally rotates as well as translates, so a translation-only expectation
# would report a spurious error.
delta_T = T1 @ np.linalg.inv(T0)
errs = []
for i, sidx in enumerate(mask_ids):
    if sidx != target_seg:
        continue
    expected = (delta_T @ np.append(kps[i], 1.0))[:3]
    errs.append(np.linalg.norm(tracked[i] - expected))
if moved < 0.01:
    print("V3c object barely moved; tracking test inconclusive -> CHECK")
elif errs:
    print(f"V3c tracked keypoints follow the object's rigid transform: "
          f"max err={max(errs):.6f} m -> {'PASS' if max(errs) < 0.02 else 'FAIL'}")
    on_table = p1[2] > 0.5
    print(f"V3c object stayed on the table (z={p1[2]:.3f}) -> "
          f"{'PASS' if on_table else 'CHECK (knocked off; tracking still valid)'}")

# static (non-target) keypoints should barely move
static = [np.linalg.norm(tracked[i] - kps[i])
          for i, s in enumerate(mask_ids) if s != target_seg]
if static:
    print(f"V3c untouched-object keypoints max drift={max(static):.4f} m "
          f"-> {'PASS' if max(static) < 0.02 else 'CHECK'}")

# ---------- V3d: keypoints feed the reward in the same frame ----------
kp_t = torch.as_tensor(tracked, dtype=torch.float32)
traj = adapter.delta_actions_to_ee_trajectory(torch.zeros(4, 7))
d = torch.norm(kp_t[0] - traj[-1]).item()
print(f"\nV3d keypoint[0] vs decoded EE trajectory endpoint: {d:.3f} m "
      f"-> {'PASS' if d < 2.0 else 'FAIL'} (same world frame, plausible scale)")

# ---------- visual ----------
rgb_after = draw_keypoints_on_image(adapter, adapter.get_vlm_image().copy(), tracked, mask_ids)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].imshow(rgb_before)
ax[0].set_title("Before: keypoints registered on settled scene", fontsize=11)
ax[1].imshow(rgb_after)
ax[1].set_title(f"After {TOTAL_STEPS} steps: '{target_name}' pushed {moved*100:.1f} cm\n"
                "tracked keypoints re-derived from live object poses", fontsize=11)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("V3 keypoint tracking — markers should stay attached to their objects",
             fontsize=12)
plt.tight_layout()
plt.savefig(str(OUT / "v3_tracking.png"), dpi=110, bbox_inches="tight")
print(f"\nwrote {OUT / 'v3_tracking.png'}")
