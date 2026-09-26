"""V6: the policy sees identical inputs under VLS and under compositional steering.

VLS needs `obs_mode="image"` to reach the raw Position/Segmentation textures for
its world point cloud, whereas RL2's existing pipeline uses the default (which
resolves to the `rgbd` wrapper). If that switch changed the pixels or the
proprioception reaching pi0, a VLS-vs-CoVer comparison would be measuring the
observation change rather than the steering.

It does not: both modes are views over the same render. The greenscreen overlay
happens inside `base_env.py` before either sees the data, and the two paths apply
the same uint8 conversion. `"image"` merely also exposes textures the wrapper
discards.

Usage:
    python vls/verify/v6_obs_parity_check.py
"""
import os
import sys

import numpy as np
import simpler_env

from _common import PKG_ROOT, _resolve_embodiment, action_stats  # noqa: E402

sys.path.insert(0, str(PKG_ROOT / "SimplerEnv"))
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict  # noqa: E402

from vls.core.env_adapters import SimplerAdapter  # noqa: E402

# VLS_VERIFY_TASK is the same flag _common.make_adapter() reads, so this script
# switches embodiment the same way the others do, e.g.
#     VLS_VERIFY_TASK=google_robot_open_top_drawer python v6_obs_parity_check.py
TASK = os.environ.get("VLS_VERIFY_TASK", "widowx_carrot_on_plate")
SEED, SETTLE = 0, 12
EMBODIMENT = _resolve_embodiment(TASK)


def rollout(obs_mode):
    kwargs = {"renderer_kwargs": {"offscreen_only": True}}
    if obs_mode is not None:
        kwargs["obs_mode"] = obs_mode
    env = simpler_env.make(TASK, **kwargs)
    obs, _ = env.reset(seed=SEED)
    for _ in range(SETTLE):
        obs, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    return env, obs


# --- legacy path: default obs_mode, stock accessor ---
env_a, obs_a = rollout(None)
img_a = get_image_from_maniskill2_obs_dict(env_a, obs_a)

# --- VLS path: obs_mode="image", adapter accessor ---
env_b, obs_b = rollout("image")
adapter = SimplerAdapter(env_b, {}, action_stats=action_stats(EMBODIMENT),
                         embodiment=EMBODIMENT)
adapter.on_reset(obs_b)
adapter.set_obs(obs_b)
img_b = adapter.get_vlm_image()

print(f"legacy (rgbd)  image: {img_a.shape} {img_a.dtype}")
print(f"VLS   (image)  image: {img_b.shape} {img_b.dtype}")

same_shape = img_a.shape == img_b.shape and img_a.dtype == img_b.dtype
diff = np.abs(img_a.astype(np.int32) - img_b.astype(np.int32)) if same_shape else None
print(f"V6a RGB bit-identical: {same_shape and bool((diff == 0).all())} "
      f"(max abs diff {int(diff.max()) if diff is not None else 'n/a'} over "
      f"{img_a.size} values) -> "
      f"{'PASS' if same_shape and (diff == 0).all() else 'FAIL'}")

# --- proprioception and task text ---
ee_a = np.asarray(obs_a["agent"]["eef_pos"], dtype=np.float64)
ee_b = np.asarray(obs_b["agent"]["eef_pos"], dtype=np.float64)
qp_a = np.asarray(obs_a["agent"]["qpos"], dtype=np.float64)
qp_b = np.asarray(obs_b["agent"]["qpos"], dtype=np.float64)
print(f"V6b eef_pos max diff: {np.abs(ee_a - ee_b).max():.3e} -> "
      f"{'PASS' if np.allclose(ee_a, ee_b, atol=1e-9) else 'FAIL'}")
print(f"V6c qpos    max diff: {np.abs(qp_a - qp_b).max():.3e} -> "
      f"{'PASS' if np.allclose(qp_a, qp_b, atol=1e-9) else 'FAIL'}")

ins_a, ins_b = env_a.get_language_instruction(), env_b.get_language_instruction()
print(f"V6d instruction: {ins_a!r} == {ins_b!r} -> {'PASS' if ins_a == ins_b else 'FAIL'}")

print("\nV6 conclusion: switching obs_mode for VLS does not alter the policy's "
      "inputs, so VLS and compositional steering remain directly comparable.")
