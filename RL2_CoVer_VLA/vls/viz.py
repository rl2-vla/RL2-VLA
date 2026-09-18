"""
Per-step rollout overlay, ported from upstream VLS `main.py:627-679`.

Produces the same annotated frames VLS writes during `bash run_main.sh`:

  * tracked keypoints with their indices, matching the numbering the VLM saw
  * a HUD: Step / Stage / Guide ON|OFF / Grip, and when guidance is active
    Norm_R / Sig_Str / Scale

Instead of drawing the sampled trajectory, the overlay draws the TWO TERMS that
are summed to form the denoising velocity at each guided step:

    v = v_pi0 + (-scale * guidance_grad)

as arrows from the current end-effector, projected with a single shared gain so
their drawn lengths are in true proportion. That makes the competition visible:
a longer yellow arrow means VLM guidance is dominating pi0's own prediction at
that step, and the legend prints both magnitudes in cm.
"""

from typing import List, Optional

import numpy as np

from vls.utils.logging_utils import SteerLogger
from vls.utils.vis_utils import add_text_to_image, draw_keypoints_on_image

log = SteerLogger("VLSViz")

# Colours (BGR, as OpenCV expects)
_PI0_BGR = (40, 170, 255)    # cyan-blue - pi0's own denoising velocity
_VLS_BGR = (255, 200, 60)    # orange    - VLM guidance term
# Arrows are normalised PER FRAME: the larger term is drawn at _REF_PX and the
# other scaled by the true ratio, so relative lengths stay exact while both stay
# visible. Real terms span ~0.05-3 cm, far too wide a range for a fixed
# px-per-metre gain. Absolute magnitudes are printed in the legend.
_REF_PX = 90.0                # on-screen length of the LARGER arrow each frame


def draw_velocity_terms(adapter, image: np.ndarray, terms: dict) -> np.ndarray:
    """Draw the two velocity terms that SUM to the denoising step.

        v = v_pi0 + (-scale * guidance_grad)

    Both are world-frame displacements in the same units, so they are projected
    from a common origin (the current EE position) with a single shared gain:
    the ratio of the drawn lengths is the true ratio of the terms. A longer
    yellow arrow means guidance dominates pi0 at that step.
    """
    import cv2

    img = np.ascontiguousarray(image)
    cam = adapter.get_camera_params()
    K, E = np.asarray(cam.intrinsic), np.asarray(cam.extrinsic)
    origin = np.asarray(adapter.get_ee_pose_world().position, dtype=np.float64)

    def project(p):
        c = (np.r_[p, 1.0] @ E.T)[:3]
        if c[2] <= 1e-6:
            return None
        uv = c @ K.T
        return int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))

    o2 = project(origin)
    if o2 is None:
        return img

    # Pixel length must be proportional to the term's true magnitude. Projecting
    # origin+vec does NOT give that: perspective foreshortens a vector pointing
    # across the view and lengthens one pointing at the camera, so a 3:1 pair can
    # draw as 4.35:1. Instead take each arrow's DIRECTION from the projection and
    # set its LENGTH from the world magnitude with one shared gain.
    o2f = np.array(o2, dtype=np.float64)
    # Only keys actually present: "vls" is absent on unguided chunks, so no
    # guidance arrow and no guidance legend row is drawn for them.
    keys = [k for k in ("pi0", "vls") if k in terms]
    if not keys:
        return img
    mags = {k: float(np.linalg.norm(np.asarray(terms[k], dtype=np.float64))) for k in keys}
    # Per-frame normalisation: the LARGER term is always drawn at _REF_PX and
    # the other is scaled by the true ratio. A fixed px-per-metre gain cannot
    # work here -- real terms span roughly 0.05-3 cm, so any single gain either
    # makes small steps invisible or sends large ones off-frame. Absolute
    # magnitudes stay readable in the legend; the arrows carry the comparison.
    biggest = max(mags.values())
    if biggest < 1e-9:
        return img
    gain = _REF_PX / biggest

    drawn = {}
    for key in keys:
        vec = np.asarray(terms[key], dtype=np.float64)
        if mags[key] < 1e-9:
            drawn[key] = None
            continue
        tip3 = project(origin + vec)       # direction only
        if tip3 is None:
            drawn[key] = None
            continue
        d = np.array(tip3, dtype=np.float64) - o2f
        n = np.linalg.norm(d)
        if n < 1e-6:
            drawn[key] = None
            continue
        end = o2f + (d / n) * (mags[key] * gain)
        drawn[key] = (int(round(end[0])), int(round(end[1])))

    for key in keys:
        colour = _PI0_BGR if key == "pi0" else _VLS_BGR
        tip = drawn.get(key)
        if tip is None or (abs(tip[0] - o2[0]) < 2 and abs(tip[1] - o2[1]) < 2):
            continue
        cv2.arrowedLine(img, o2, tip, (0, 0, 0), 6, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(img, o2, tip, colour, 3, cv2.LINE_AA, tipLength=0.25)

    cv2.circle(img, o2, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, o2, 5, (0, 0, 0), 1, cv2.LINE_AA)

    # Small legend: colour swatch + magnitude, so the arrows are identifiable
    # and the picture stays quantitative.
    h = img.shape[0]
    labels = {"pi0": "pi0", "vls": "VLS"}
    colours = {"pi0": _PI0_BGR, "vls": _VLS_BGR}
    rows = [(labels[k], colours[k], mags[k]) for k in keys]
    box_h = 14 * len(rows) + 8
    cv2.rectangle(img, (6, h - box_h - 6), (132, h - 4), (0, 0, 0), -1)
    for i, (lbl, colour, mag) in enumerate(rows):
        y = h - box_h + 6 + i * 14
        cv2.line(img, (12, y), (28, y), colour, 3, cv2.LINE_AA)
        cv2.putText(img, f"{lbl} {mag*100:4.2f}cm", (34, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_frame(
    adapter,
    terms: Optional[dict] = None,
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
        terms: {"pi0","vls"} world-frame displacements of the two summed
            velocity terms, from GuidedSampler.get_last_terms().
        keypoints / mask_ids: current tracked keypoints, drawn with their indices.
        normalized_reward / guide_scale: from the sampler, for the HUD.
    """
    image = np.array(adapter.get_vlm_image())

    if draw_trajectory and terms is not None:
        try:
            image = draw_velocity_terms(adapter, image, terms)
        except Exception as e:                      # never kill a rollout for a picture
            log.warning(f"velocity-term overlay failed: {e}")

    if draw_keypoints and keypoints is not None and len(keypoints):
        try:
            image = draw_keypoints_on_image(
                adapter=adapter, image=image, keypoints=keypoints, mask_ids=mask_ids
            )
        except Exception as e:
            log.warning(f"keypoint overlay failed: {e}")

    # BridgeSimplerAdapter.postprocess_gripper: -1=CLOSE, +1=OPEN on the
    # executed action channel (INT-ACT/.../simpler.py:218-224). Must match the
    # polarity used by the Schmitt trigger in steering.py's _update_stage.
    gripper_str = (
        f"Grip:{'O' if gripper_val is not None and gripper_val > 0 else 'C'}"
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
