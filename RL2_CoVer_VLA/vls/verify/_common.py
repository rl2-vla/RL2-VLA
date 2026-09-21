"""Shared setup for the VLS port verification scripts.

Resolves repo-relative paths so the scripts run from anywhere, and builds a
configured SimplerAdapter against a live SIMPLER env.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

# .../RL2-VLA/RL2_CoVer_VLA/vls/verify/_common.py -> RL2_CoVer_VLA, RL2-VLA
PKG_ROOT = Path(__file__).resolve().parents[2]      # RL2_CoVer_VLA
REPO_ROOT = PKG_ROOT.parent                          # RL2-VLA
BRIDGE_STATS = REPO_ROOT / "INT-ACT/config/dataset/bridge_statistics.json"
DEFAULT_OUT = REPO_ROOT / "outputs/vls_verify"

if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))


# Keypoint-detector settings, matching upstream VLS (configs/perception.yaml)
# except for three values (bounds, merge radius, feature extractor) noted below.
# Detection is fully local (DINO features + k-means) and needs no API key.
# KEEP IN SYNC with KEYPOINT_DETECTOR_CFG in vls/runtime.py, which is what the
# eval actually uses -- these scripts should verify the same detector.
#
#   bounds_*                 SIMPLER-specific. CALVIN's box
#                            ([-1,-0.75,-0.1]..[0.1,0.75,1.2]) rejects every
#                            point here, since SIMPLER's table sits at z ~ 0.87.
#   min_dist_bt_keypoints    MeanShift merge radius; sets keypoint DENSITY (max 5
#                            candidates per object). VLS uses 0.05, which
#                            collapses SIMPLER's 3-20 cm objects to ~1 keypoint
#                            each (mean/object over 5 scenes/task: 1.35 at 0.05,
#                            2.1 at 0.03, 2.7 at 0.025, 4.5 at 0.015). 0.025
#                            targets ~2-3 per object -- an ESTIMATE of upstream's
#                            density -- and avoids 0.015's 5-deep label pile-up
#                            on 3 cm cubes.
#   feature_extractor        VLS uses dinov3_vitb16; we use dinov2_vitb14, which
#                            matches its capacity (both 768-dim ViT-B features).
#                            DINOv3 is unreachable here for two INDEPENDENT
#                            reasons, neither worked around by relaxing pins:
#                              1. It needs transformers>=4.56, and 4.56 BREAKS
#                                 pi0 -- denoise_step() raises "AttributeError:
#                                 'GemmaModel' object has no attribute 'model'"
#                                 because the lerobot fork depends on Gemma
#                                 internals that changed. (">=4.56.0" also
#                                 resolves to 5.17.0, which is worse.) We stay
#                                 on transformers==4.48.3.
#                              2. In 4.48 the DINOv3ViTModel class does not
#                                 exist at all, so there is nothing to load the
#                                 weights into -- ignoring the requirement does
#                                 not help.
#                            torch.hub's facebookresearch/dinov3 sidesteps
#                            transformers entirely and loads, but Meta ships
#                            only HF-format weights (211 tensors, embeddings.*
#                            naming) vs the hub model's 188 (cls_token naming),
#                            with 2 keys overlapping -- a conversion would be
#                            unverifiable on the critical path.
#                            keypoint_detector.py:148 silently falls back to
#                            dinov2_vits14 on ANY load failure, so make_detector()
#                            raises rather than let that pass unnoticed.
KEYPOINT_DETECTOR_CFG = {
    "num_candidates_per_mask": 5,          # VLS default
    "min_dist_bt_keypoints": 0.025,        # VLS 0.05 -> see note above
    "max_mask_ratio": 0.5,                 # VLS default
    "feature_extractor": "dinov2_vitb14",  # VLS dinov3_vitb16 -> see note above
    "device": "cuda",
    "seed": 0,
    "bounds_min": [-1.0, -1.0, 0.80],      # SIMPLER tabletop workspace
    "bounds_max": [1.0, 1.0, 1.30],
}


def action_stats() -> dict:
    """Bridge p01/p99 action statistics used by the trajectory decoder."""
    stats = json.loads(BRIDGE_STATS.read_text())["action"]
    return {"p01": stats["p01"], "p99": stats["p99"]}


def make_detector(**overrides):
    """VLS KeypointDetector configured for SIMPLER (no API key required).

    Verifies the extractor that actually loaded matches the one requested.
    `keypoint_detector.py:148` catches *any* load failure and quietly drops to
    dinov2_vits14, so without this check a gated/misspelled model would run as a
    different network while the config claims otherwise.
    """
    from vls.core.keypoint_detector import KeypointDetector

    cfg = dict(KEYPOINT_DETECTOR_CFG)
    cfg.update(overrides)
    detector = KeypointDetector(cfg)

    requested = cfg["feature_extractor"]
    actual_family = getattr(detector, "feature_extractor_type", "?")
    if not requested.startswith(actual_family):
        raise RuntimeError(
            f"Keypoint detector silently fell back: requested {requested!r} but "
            f"loaded a {actual_family!r} model. If {requested!r} is a gated "
            f"HuggingFace repo, access is still pending; pin an available model "
            f"in KEYPOINT_DETECTOR_CFG instead of letting the fallback hide it."
        )
    return detector


def make_adapter(task: str = None, seed: int = 0, settle: int = 12):
    """Build a SIMPLER env + SimplerAdapter, settled past the reset transient.

    `task` defaults to $VLS_VERIFY_TASK, else widowx_carrot_on_plate, so every
    verify script can be pointed at another task without editing it, e.g.
        VLS_VERIFY_TASK=widowx_stack_cube python v3_keypoint_check.py <out_dir>

    Returns (env, adapter, obs).
    """
    task = task or os.environ.get("VLS_VERIFY_TASK", "widowx_carrot_on_plate")
    import simpler_env
    from vls.core.env_adapters import SimplerAdapter

    env = simpler_env.make(task, obs_mode="image",
                           renderer_kwargs={"offscreen_only": True})
    obs, _ = env.reset(seed=seed)

    adapter = SimplerAdapter(env, {"vlm_camera": "3rd_view_camera"},
                             action_stats=action_stats())
    adapter.on_reset(obs)

    # Objects are still settling right after reset; keypoints registered then
    # would attach to mid-fall poses.
    for _ in range(settle):
        obs, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    adapter.set_obs(obs)

    return env, adapter, obs
