"""
Per-step rollout overlay, ported from upstream VLS `main.py:627-679`.

Produces the same annotated frames VLS writes during `bash run_main.sh`:

  * the sampled action chunks projected into the camera as 3D trajectories
    (one coloured polyline per particle, so the count tracks sample_batch_size)
  * tracked keypoints with their indices, matching the numbering the VLM saw
  * a HUD: Step / Stage / Guide ON|OFF / Grip, and when guidance is active
    Norm_R / Sig_Str / Scale

This is not decoration. `draw_action_trajectory_on_vlm_image` renders the
polyline by calling `adapter.delta_actions_to_ee_trajectory` and then
`project_3d_to_2d` -- the same decoder the guidance reward is computed on. A
frame/sign/scale error shows up as a polyline that drifts off the gripper or
bends the wrong way, which is far quicker to spot than in logs.
"""

from typing import List, Optional

import numpy as np

from vls.utils.logging_utils import SteerLogger
from vls.utils.vis_utils import (
    add_text_to_image,
    draw_action_trajectory_on_vlm_image,
    draw_keypoints_on_image,
)

log = SteerLogger("VLSViz")


def render_frame(
    adapter,
    action_chunk=None,
    keypoints: Optional[np.ndarray] = None,
    mask_ids: Optional[np.ndarray] = None,
    *,
    global_step: int = 0,
    current_stage: int = 1,
    use_guidance: bool = False,
    gripper_val: Optional[float] = None,
    normalized_reward: Optional[float] = None,
    guide_scale: Optional[float] = None,
    sigmoid_k: float = 25.0,
    sigmoid_x0: float = 0.75,
    action_horizon: int = 4,
    draw_trajectory: bool = True,
    draw_keypoints: bool = True,
) -> np.ndarray:
    """Build one annotated frame. Mirrors upstream main.py:627-679.

    Args:
        action_chunk: (B, T, D) sampled actions; every particle is drawn, so the
            number of polylines follows sample_batch_size.
        keypoints / mask_ids: current tracked keypoints, drawn with their indices.
        normalized_reward / guide_scale: from the sampler, for the HUD.
    """
    image = None
    if draw_trajectory and action_chunk is not None:
        try:
            image = draw_action_trajectory_on_vlm_image(
                adapter=adapter,
                action_chunk=action_chunk,
                num_steps=action_horizon,
                global_step=global_step,
                action_executed=0,
            )
        except Exception as e:                      # never kill a rollout for a picture
            log.warning(f"trajectory overlay failed: {e}")
    if image is None:
        image = np.array(adapter.get_vlm_image())

    if draw_keypoints and keypoints is not None and len(keypoints):
        try:
            image = draw_keypoints_on_image(
                adapter=adapter, image=image, keypoints=keypoints, mask_ids=mask_ids
            )
        except Exception as e:
            log.warning(f"keypoint overlay failed: {e}")

    gripper_str = (
        f"Grip:{'O' if gripper_val is not None and gripper_val < 0 else 'C'}"
        f"({gripper_val:.2f})"
        if gripper_val is not None
        else "Grip:-"
    )
    status: List[str] = [
        f"Step:{global_step} Stage:{current_stage}",
        f"Guide:{'ON' if use_guidance else 'OFF'} {gripper_str}",
    ]

    if use_guidance and normalized_reward is not None:
        norm_r = float(normalized_reward)
        strength = 1.0 / (1.0 + np.exp(sigmoid_k * (norm_r - sigmoid_x0)))
        scale_str = (
            f"{guide_scale:.1f}" if guide_scale is not None and guide_scale > 0 else "-"
        )
        status.extend([
            f"Norm_R: {norm_r:.2f}",
            f"Sig_Str: {strength:.1%}",
            f"Scale: {scale_str}",
        ])

    try:
        return add_text_to_image(image, status)
    except Exception as e:
        log.warning(f"HUD overlay failed: {e}")
        return image
