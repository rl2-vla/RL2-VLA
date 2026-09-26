"""V1 (point cloud geometry) + V2 (segmentation) verification for SimplerAdapter.

Usage:
    python vls/verify/v1_v2_check.py
"""
import numpy as np

from _common import make_adapter  # noqa: E402

env, adapter, obs = make_adapter()

rgb, depth, points, seg, names = adapter.get_keypoint_detection_inputs()
cp = adapter.get_camera_params()
H, W = depth.shape
print(f"rgb {rgb.shape} {rgb.dtype} | depth {depth.shape} | points {points.shape} | seg {seg.shape}")

# ---------- V1d: non-mutation ----------
raw_before = np.array(obs["image"][adapter.vlm_camera]["Position"], copy=True)
_, _, points2, _, _ = adapter.get_keypoint_detection_inputs()
raw_after = np.asarray(obs["image"][adapter.vlm_camera]["Position"])
print(f"V1d non-mutation: Position unchanged={np.array_equal(raw_before, raw_after)} | "
      f"clouds identical={np.array_equal(points, points2)}")

# ---------- V1a: reprojection round-trip ----------
valid = np.linalg.norm(points, axis=-1) > 1e-6
vs, us = np.nonzero(valid)
idx = np.random.default_rng(0).choice(len(vs), size=min(500, len(vs)), replace=False)
vs, us = vs[idx], us[idx]
P = points[vs, us]                                   # (N,3) world
K, E = cp.intrinsic, cp.extrinsic                    # OpenCV convention
cam = (np.c_[P, np.ones(len(P))] @ E.T)[:, :3]
proj = cam @ K.T
uv = proj[:, :2] / proj[:, 2:3]
err = np.linalg.norm(uv - np.c_[us, vs], axis=1)
print(f"V1a reprojection px err: median={np.median(err):.4f} p99={np.percentile(err,99):.4f} "
      f"-> {'PASS' if np.median(err) < 1 and np.percentile(err,99) < 2 else 'FAIL'}")
# vertical-mirror signature check (the LIBERO flipud trap)
mirror = np.linalg.norm(uv - np.c_[us, H - 1 - vs], axis=1)
print(f"    mirror-hypothesis median err={np.median(mirror):.2f} (should be >> above if no flip)")

# ---------- V1b: TCP anchoring ----------
# Measure against the gripper FINGER geometry, not the pixel the TCP projects
# onto: the TCP is the midpoint *between* the fingers, so that ray passes
# through empty space to whatever is behind (here the arena wall, ~24cm away).
# Comparing to the finger surfaces is the meaningful check.
tcp = adapter.get_ee_pose_world().position.astype(np.float64)
c = (np.r_[tcp, 1.0] @ E.T)[:3]
p2 = c @ K.T
u, v = int(round(p2[0] / p2[2])), int(round(p2[1] / p2[2]))

links = {l.id: l for l in env.unwrapped.agent.robot.get_links()}
finger_ids = [i for i, l in links.items() if "finger" in l.name]
actor_seg_raw = obs["image"][adapter.vlm_camera]["Segmentation"][..., 1]
fmask = np.isin(actor_seg_raw, finger_ids) & valid
if fmask.sum() > 20:
    fd = np.linalg.norm(points[fmask] - tcp, axis=1)
    print(f"V1b TCP anchor: tcp={np.round(tcp,3)} px=({u},{v}) | finger px={fmask.sum()} "
          f"min|cloud-tcp|={fd.min():.4f} m -> {'PASS' if fd.min() < 0.02 else 'FAIL'}")
    for i in finger_ids:
        mm = (actor_seg_raw == i) & valid
        if mm.sum() > 20:
            off = np.linalg.norm(points[mm].mean(0) - np.asarray(links[i].pose.p))
            print(f"    {links[i].name:<20} n={mm.sum():6d} |centroid-linkpose|={off:.4f} m")
else:
    print("V1b TCP anchor: gripper not visible in frame (skipped)")

# ---------- V1c: table plane / world axes ----------
zs = points[valid][:, 2]
hist, edges = np.histogram(zs, bins=80)
zmode = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
band = points[valid][np.abs(zs - zmode) < 0.01]
if len(band) > 200:
    c0 = band.mean(0)
    _, _, Vt = np.linalg.svd(band - c0, full_matrices=False)
    n = Vt[-1] / np.linalg.norm(Vt[-1])
    if n[2] < 0:
        n = -n
    rms = np.sqrt((((band - c0) @ n) ** 2).mean())
    ang = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
    print(f"V1c dominant plane: z≈{zmode:.3f} normal={np.round(n,3)} "
          f"angle_from_+Z={ang:.2f}° rms={rms*1000:.2f} mm "
          f"-> {'PASS' if ang < 2 and rms < 0.005 else 'CHECK'}")

# ---------- V2: segmentation ----------
print(f"\nV2 segments ({len(names)-1} objects): "
      + ", ".join(f"{k}:{v}" for k, v in sorted(names.items()) if k != 0))
robot_ids = {l.id for l in env.unwrapped.agent.robot.get_links()}
actor_seg = obs["image"][adapter.vlm_camera]["Segmentation"][..., 1]
leak = set(np.unique(actor_seg[seg > 0])) & robot_ids
print(f"V2b robot-link leakage into labelled segments: {leak or 'none'} "
      f"-> {'PASS' if not leak else 'FAIL'}")

print("V2c pose vs point-cloud centroid:")
for sidx, nm in sorted(names.items()):
    if sidx == 0:
        continue
    m = (seg == sidx) & valid
    if m.sum() < 30:
        continue
    cen = points[m].mean(0)
    pose = adapter.get_object_pose_by_segment(sidx).position
    print(f"    [{sidx}] {nm:<28} n={m.sum():6d} |centroid-pose|={np.linalg.norm(cen-pose):.4f} m")
