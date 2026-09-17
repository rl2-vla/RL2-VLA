"""V3b: keypoint tracking is unaffected by visual occlusion.

Upstream VLS's README describes the keypoint tracker as using "optical flow"
with "occlusion handling and reinitialization". That is not what the code does:
there is no optical flow anywhere in the repo. `KeypointTracker` stores each
keypoint as an offset in its object's LOCAL frame at registration time, and
`get_keypoint_positions()` re-derives world positions from
`adapter.get_object_pose_by_segment()` — the ground-truth simulator pose. The
image is never consulted after registration.

The practical consequence matters for interpreting the V3 visual: when a tracked
object leaves the frame or is hidden behind the gripper, its keypoint markers
disappear from the *picture* while the underlying 3D positions remain exact. The
guidance reward, which consumes the 3D positions, is unaffected.

This script drives the gripper directly over the target object to occlude it and
shows tracking error stays at zero throughout.

Usage:
    python vls/verify/v3b_occlusion_check.py [output_dir]

Writes v3b_occlusion.png: frames at increasing occlusion with the tracked
keypoint projected on top, plus visible-pixel count and tracking error per frame.
"""
import sys
from pathlib import Path

import numpy as np

from _common import DEFAULT_OUT, make_adapter, make_detector  # noqa: E402

from vls.core.keypoint_tracker import KeypointTracker  # noqa: E402
from vls.utils.vis_utils import draw_keypoints_on_image  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
OUT.mkdir(parents=True, exist_ok=True)

env, adapter, obs = make_adapter()
KEYPOINT_DETECTOR = make_detector()

# ---------- no optical flow exists ----------
import inspect  # noqa: E402

src = inspect.getsource(KeypointTracker)
flow_terms = ["opticalflow", "calcopticalflow", "cotracker", "raft", "tapir", "tapnet"]
found = [t for t in flow_terms if t in src.lower().replace("_", "").replace("-", "")]
uses_pose = "get_object_pose_by_segment" in src
print(f"V3b0 optical-flow machinery in KeypointTracker: {found or 'none'} "
      f"-> {'PASS' if not found else 'FAIL'}")
print(f"V3b0 tracks via adapter.get_object_pose_by_segment (sim pose): {uses_pose} "
      f"-> {'PASS' if uses_pose else 'FAIL'}")

# ---------- register keypoints on the target with the REAL VLS detector ----------
# Same call path as upstream main.py:322 -- local DINO features + k-means per
# mask. No API key required for detection.
rgb, _, points, seg, names = adapter.get_keypoint_detection_inputs()
target_seg = sorted(k for k in names if k != 0)[0]

all_kps, _, all_mask_ids = KEYPOINT_DETECTOR.get_keypoints(rgb, points, seg, names)
sel = all_mask_ids == target_seg
kps = all_kps[sel].astype(np.float32)
mask_ids = all_mask_ids[sel].astype(np.int32)
if len(kps) == 0:
    raise RuntimeError(f"Detector found no keypoints on segment {target_seg}")

tracker = KeypointTracker(adapter)
tracker.register_keypoints(kps, mask_ids, names)
T0 = adapter.get_object_pose_by_segment(target_seg).to_transformation_matrix()
print(f"\nV3b tracking '{names[target_seg]}' via {len(kps)} keypoints "
      f"({KEYPOINT_DETECTOR.feature_extractor_type} + k-means)")

# ---------- drive the gripper over the object to occlude it ----------
# Hover only: the object must NOT move, so any tracking error would come purely
# from the occlusion rather than from motion.
# Park the gripper on the LINE OF SIGHT between camera and object, partway
# along it. That blocks the view without the gripper ever reaching the object,
# so occlusion is isolated from contact: the object must not move at all.
STEP_M, N_STEPS = 0.02, 55
cam_params = adapter.get_camera_params()
cam_world = np.linalg.inv(cam_params.extrinsic)[:3, 3]      # camera centre in world
obj_world = adapter.get_object_pose_by_segment(target_seg).position.copy()
# 22cm from the object toward the camera: in front of it, but clear of it
los_dir = (cam_world - obj_world) / (np.linalg.norm(cam_world - obj_world) + 1e-9)
block_point = obj_world + los_dir * 0.22

R_base = adapter.get_robot_base_pose().to_rotation_matrix()
frames, visible_px, errors = [], [], []
snap_at = {0, 18, 36, N_STEPS - 1}

for step in range(N_STEPS):
    goal = block_point - adapter.get_ee_pose_world().position
    d = R_base.T @ goal
    n = np.linalg.norm(d) + 1e-6
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    act[:3] = d / n * min(STEP_M, n)
    act[-1] = -1.0
    obs, *_ = env.step(act)
    adapter.set_obs(obs)

    seg_now = adapter.process_segmentation(obs["image"][adapter.vlm_camera]["Segmentation"])[0]
    npx = int((seg_now == target_seg).sum())
    T1 = adapter.get_object_pose_by_segment(target_seg).to_transformation_matrix()
    # full rigid transform since registration (identity here, as we only hover)
    delta_T = T1 @ np.linalg.inv(T0)
    tracked_now = tracker.get_keypoint_positions()
    err = max(
        float(np.linalg.norm(tracked_now[i] - (delta_T @ np.append(kps[i], 1.0))[:3]))
        for i in range(len(kps))
    )
    visible_px.append(npx)
    errors.append(err)

    if step in snap_at:
        img = draw_keypoints_on_image(
            adapter, adapter.get_vlm_image().copy(),
            tracker.get_keypoint_positions(), mask_ids
        )
        frames.append((step, npx, err, img))

moved = float(np.linalg.norm(
    adapter.get_object_pose_by_segment(target_seg).position - T0[:3, 3]))
print(f"V3b object displacement during hover: {moved*100:.3f} cm "
      f"-> {'PASS' if moved < 0.01 else 'CHECK'} (should be ~0: we only hovered)")
print(f"V3b visible pixels: {visible_px[0]} -> {min(visible_px)} "
      f"(occlusion {100*(1-min(visible_px)/max(visible_px[0],1)):.0f}% at peak)")
print(f"V3b max tracking error across all {N_STEPS} frames: {max(errors):.6f} m "
      f"-> {'PASS' if max(errors) < 1e-3 else 'FAIL'}")
print("V3b conclusion: tracking error is independent of visibility — the tracker "
      "reads sim pose, not pixels.")

# ---------- visual ----------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(2, len(frames), height_ratios=[3, 2])
for i, (step, npx, err, img) in enumerate(frames):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"step {step}\nvisible px={npx}\nerr={err:.6f} m", fontsize=9)

axb = fig.add_subplot(gs[1, :])
axb.plot(visible_px, color="#c2410c", label="visible pixels of target")
axb.set_ylabel("visible pixels", color="#c2410c")
axb.tick_params(axis="y", labelcolor="#c2410c")
axb.set_xlabel("step")
ax2 = axb.twinx()
ax2.plot(errors, color="#0b6fa4", label="tracking error (m)")
ax2.set_ylabel("tracking error (m)", color="#0b6fa4")
ax2.tick_params(axis="y", labelcolor="#0b6fa4")
ax2.set_ylim(-1e-3, 1e-3)
axb.grid(alpha=0.3)
axb.set_title("Visibility collapses as the gripper moves over the object; "
              "tracking error stays flat at zero", fontsize=10)

fig.suptitle("V3b — keypoint tracking is pose-based, not vision-based: "
             "occlusion does not degrade it", fontsize=12)
plt.tight_layout()
plt.savefig(str(OUT / "v3b_occlusion.png"), dpi=110, bbox_inches="tight")
print(f"\nwrote {OUT / 'v3b_occlusion.png'}")
