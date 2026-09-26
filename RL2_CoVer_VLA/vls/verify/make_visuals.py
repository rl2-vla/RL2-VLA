"""Render verification visuals: RGB, depth, segmentation, world-Z panel + PLY.

Usage:
    python vls/verify/make_visuals.py [output_dir]

Writes v1_v2_panel.png and scene.ply (default: RL2-VLA/outputs/vls_verify/).
"""
import sys
from pathlib import Path

import numpy as np

from _common import DEFAULT_OUT, make_adapter  # noqa: E402

from vls.utils.vis_utils import save_pointcloud_with_keypoints_ply  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
OUT.mkdir(parents=True, exist_ok=True)

env, adapter, obs = make_adapter()

rgb, depth, points, seg, names = adapter.get_keypoint_detection_inputs()
cp = adapter.get_camera_params()
H, W = depth.shape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(2, 2, figsize=(14, 10))

ax[0, 0].imshow(rgb)
ax[0, 0].set_title("RGB (greenscreened composite)\nwhat the VLM sees", fontsize=11)

d = np.where(depth > 0, depth, np.nan)
im = ax[0, 1].imshow(d, cmap="viridis")
ax[0, 1].set_title(f"Depth (m)  range {np.nanmin(d):.2f}–{np.nanmax(d):.2f}", fontsize=11)
plt.colorbar(im, ax=ax[0, 1], fraction=0.035)

# segmentation: background grey, each object a distinct colour
segvis = np.zeros((H, W, 3))
segvis[seg == 0] = [0.85, 0.85, 0.85]
# 6 colours: covers WidowX's usual 1-3 objects and Google Robot's cabinet
# scenes (body + up to 3 drawers, + apple = 5 for apple-in-drawer).
palette = [
    [0.90, 0.25, 0.20], [0.20, 0.45, 0.90], [0.20, 0.75, 0.35],
    [0.95, 0.65, 0.10], [0.55, 0.30, 0.75], [0.10, 0.70, 0.70],
]
for i, sidx in enumerate(sorted(k for k in names if k != 0)):
    segvis[seg == sidx] = palette[i % len(palette)]
# overlay robot pixels in dark grey to show they are EXCLUDED from labels
robot_ids = {l.id for l in env.unwrapped.agent.robot.get_links()}
robot_mask = np.isin(obs["image"][adapter.vlm_camera]["Segmentation"][..., 1], list(robot_ids))
segvis[robot_mask & (seg == 0)] = [0.35, 0.35, 0.35]
ax[1, 0].imshow(segvis)
# Strip WidowX's literal "bridge_" prefix specifically (-> "carrot", "plate"),
# rather than "drop whichever token comes first": that generic rule also ate
# Google Robot's FIRST token, which is the informative one ("top_drawer" /
# "middle_drawer" / "bottom_drawer" all collapsed to "drawer").
def _short_name(name):
    return name[len("bridge_"):].split("_")[0] if name.startswith("bridge_") else name


lbl = "  |  ".join(f"{k}={_short_name(names[k])}" for k in sorted(names) if k != 0)
ax[1, 0].set_title(f"Segmentation (actor-level)\n{lbl}   [dark grey = robot, excluded]", fontsize=10)

# world-Z heatmap proves the plane fit / axis convention
z = np.where(np.linalg.norm(points, axis=-1) > 1e-6, points[..., 2], np.nan)
im2 = ax[1, 1].imshow(z, cmap="coolwarm")
ax[1, 1].set_title("World Z of each pixel (m)\nflat table => uniform colour, +Z up", fontsize=11)
plt.colorbar(im2, ax=ax[1, 1], fraction=0.035)

# mark TCP + finger cloud
K, E = cp.intrinsic, cp.extrinsic
tcp = adapter.get_ee_pose_world().position.astype(np.float64)
c = (np.r_[tcp, 1.0] @ E.T)[:3]
p = c @ K.T
u, v = p[0] / p[2], p[1] / p[2]
for a in (ax[0, 0], ax[1, 0]):
    a.plot(u, v, "x", color="lime", markersize=14, markeredgewidth=3, label="TCP (projected)")
    a.legend(loc="upper right", fontsize=9)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])

plt.tight_layout()
plt.savefig(str(OUT / "v1_v2_panel.png"), dpi=110, bbox_inches="tight")
print("wrote v1_v2_panel.png")

# PLY with object centroids as pseudo-keypoints (uses VLS's own util)
kps = []
for sidx in sorted(k for k in names if k != 0):
    m = (seg == sidx) & (np.linalg.norm(points, axis=-1) > 1e-6)
    if m.sum() > 30:
        kps.append(points[m].mean(0))
kps = np.array(kps) if kps else np.zeros((0, 3))
save_pointcloud_with_keypoints_ply(points, rgb, kps, str(OUT / "scene.ply"))
print(f"wrote scene.ply  ({int((np.linalg.norm(points,axis=-1)>1e-6).sum())} valid pts, {len(kps)} keypoints)")
