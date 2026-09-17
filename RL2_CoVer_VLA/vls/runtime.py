"""
One-call setup for VLS inside RL2's SIMPLER rollout.

Keeps the edits in ``run_simpler_eval_with_openpi.py`` small: the rollout calls
``build_vls(...)`` once per task and gets back a controller that owns the
adapter, detector, tracker, sampler and stage machine.

Everything here is inert unless ``cfg.use_vls`` is set.
"""

import json
import os
from pathlib import Path
from typing import Optional

from vls.core.env_adapters import SimplerAdapter
from vls.core.keypoint_detector import KeypointDetector
from vls.core.keypoint_tracker import KeypointTracker
from vls.core.pi0_steer import GuidedSampler
from vls.steering import VLSSteeringController
from vls.utils.logging_utils import SteerLogger

log = SteerLogger("VLSRuntime")

# Repo roots: .../RL2-VLA/RL2_CoVer_VLA/vls/runtime.py
PKG_ROOT = Path(__file__).resolve().parents[1]     # RL2_CoVer_VLA
REPO_ROOT = PKG_ROOT.parent                        # RL2-VLA
BRIDGE_STATS = REPO_ROOT / "INT-ACT/config/dataset/bridge_statistics.json"

# Detector settings: upstream VLS (configs/perception.yaml) except where
# SIMPLER's scene scale forces a change. See vls/README.md for the rationale.
#   feature_extractor      dinov2_vitb14, not VLS's dinov3_vitb16: DINOv3 needs
#                          transformers>=4.56, which breaks pi0's Gemma path.
#                          Same 768-dim ViT-B capacity.
#   bounds_*               SIMPLER's table sits at z ~ 0.87; CALVIN's box would
#                          reject every point.
#   min_dist_bt_keypoints  Bridge objects are ~11 cm across, so VLS's 5 cm merge
#                          radius collapses each object to a single keypoint.
KEYPOINT_DETECTOR_CFG = {
    "num_candidates_per_mask": 5,
    "min_dist_bt_keypoints": 0.015,
    "max_mask_ratio": 0.5,
    "feature_extractor": "dinov2_vitb14",
    "device": "cuda",
    "seed": 0,
    "bounds_min": [-1.0, -1.0, 0.80],
    "bounds_max": [1.0, 1.0, 1.30],
}


def _action_stats() -> dict:
    stats = json.loads(BRIDGE_STATS.read_text())["action"]
    return {"p01": stats["p01"], "p99": stats["p99"]}


def _make_detector(cfg_overrides: Optional[dict] = None) -> KeypointDetector:
    """Build the detector, refusing to accept a silent extractor substitution.

    `keypoint_detector.py:148` catches any load failure and drops to
    dinov2_vits14 with only a warning, so without this check a gated or
    misspelled model would run as a different network.
    """
    cfg = dict(KEYPOINT_DETECTOR_CFG)
    cfg.update(cfg_overrides or {})
    detector = KeypointDetector(cfg)

    requested = cfg["feature_extractor"]
    actual = getattr(detector, "feature_extractor_type", "?")
    if not requested.startswith(actual):
        raise RuntimeError(
            f"Keypoint detector silently fell back: requested {requested!r} but "
            f"loaded a {actual!r} model."
        )
    return detector


def build_vls(env, cfg, pi0_policy, task_key: Optional[str] = None):
    """Construct the VLS stack for one task.

    Heavy objects (the DINO model in particular) are built once here, not per
    episode. Returns a ``VLSSteeringController``.
    """
    adapter = SimplerAdapter(
        env,
        {
            "vlm_camera": "3rd_view_camera",
            "guide_scale": cfg.vls_guide_scale,
        },
        action_stats=_action_stats(),
    )

    detector = _make_detector()
    tracker = KeypointTracker(adapter)
    sampler = GuidedSampler(pi0_policy, adapter, cfg)

    vlm_agent = None
    stage_recognizer = None
    # Convert empty string or string "None" to Python None
    guidance_dir = cfg.vls_guidance_dir
    if not guidance_dir or guidance_dir == "None":
        guidance_dir = None

    if guidance_dir is None:
        # Only needed when synthesizing guidance; cached mode is fully offline.
        from vls.vlm_query.vlm_agent import VLMAgent

        # Matches VLS's live config (configs/perception.yaml:24-30), which is
        # what a `bash run_main.sh` run actually loads. Note the ablation YAMLs
        # say gpt-5.2-chat, but those are not the runtime path.
        # gpt-4o is NOT a substitute: it ignored the prompt's output contract
        # and returned a plain image caption with no stages or code, which
        # _parse_other_metadata rejects with "num_stages not found in output".
        # env_type="simpler" selects guidance_template_simpler.txt as the
        # task-specific insert; guidance_template.txt is the shared base that
        # carries the output contract (num_stages, function format). Both are
        # required -- without the base, the prompt degrades to a 43-byte stub
        # and the VLM returns prose that _parse_other_metadata rejects.
        vlm_agent = VLMAgent(
            {
                "model": os.environ.get("VLS_OPENAI_MODEL", "gpt-5.1"),
                "temperature": 1.0,
                "max_completion_tokens": 2000,
                "query_template_dir": str(PKG_ROOT / "vls/vlm_query"),
            },
            env_type="simpler",
        )

    if cfg.vls_use_vlm_stage_recognition:
        try:
            from vls.core.gemini_grounder import create_gemini_stage_recognizer

            # Matches VLS's live config (configs/perception.yaml:44).
            stage_recognizer = create_gemini_stage_recognizer({
                "enabled": True,
                "model": os.environ.get("VLS_GEMINI_MODEL", "gemini-3-flash-preview"),
            })
        except Exception as e:
            log.warning(f"Stage recognition disabled ({e}); "
                        f"falling back to the reward/gripper Schmitt trigger alone")

    recorder = None
    if getattr(cfg, "vls_save_video", False):
        from vls.utils.vis_utils import TrajectoryVideoRecorder

        recorder = TrajectoryVideoRecorder(
            output_dir=os.path.join(cfg.local_log_dir, "vls"), fps=10
        )

    controller = VLSSteeringController(
        adapter=adapter,
        detector=detector,
        tracker=tracker,
        sampler=sampler,
        guidance_dir=guidance_dir,
        vlm_agent=vlm_agent,
        stage_recognizer=stage_recognizer,
        vlm_query_limit=cfg.vls_vlm_query_limit,
        schmitt_upper=cfg.vls_schmitt_upper,
        schmitt_lower=cfg.vls_schmitt_lower,
        default_output_dir=os.path.join(cfg.local_log_dir, "vls"),
        task_key=task_key or getattr(cfg, "task_suite_name", None),
    )
    controller.recorder = recorder          # None unless vls_save_video
    controller.cfg = cfg
    log.info(
        f"VLS ready | extractor={KEYPOINT_DETECTOR_CFG['feature_extractor']} "
        f"guide_scale={cfg.vls_guide_scale} diversity={cfg.vls_diversity_scale} "
        f"fkd={cfg.vls_use_fkd} guidance={'cached' if guidance_dir else 'VLM'}"
    )
    return controller
