"""
SIMPLER (ManiSkill2_real2sim / SAPIEN) adapter for VLS steering.

Implements the upstream ``BaseEnvAdapter`` contract against SIMPLER, so the
ported VLS components (keypoint detector/tracker, guided samplers, viz) run
unchanged.

Scope: WidowX / Bridge and Google Robot / fractal, via
``env_config["embodiment"]`` (default ``"widowx"``).

Google Robot's checkpoint normalizes actions with MEAN_STD *inside*
``PI0Policy.unnormalize_outputs`` (WidowX's is IDENTITY there; its real
denormalize happens externally in ``BridgeSimplerAdapter.postprocess``). Since
VLS steers the sample before ``unnormalize_outputs`` runs (a ``@torch.no_grad``
call it can't backprop through), ``_probe_action_affine`` measures the affine
across BOTH stages in rollout order, which is correct for either checkpoint
without embodiment-specific math -- pass the live ``policy`` for this.

Key SIMPLER facts this relies on (all verified against the vendored source):

* ``obs["image"][cam]`` -> {"Color", "Position", "Segmentation"} under
  ``obs_mode="image"``; ``obs["camera_param"][cam]`` -> {"intrinsic_cv",
  "extrinsic_cv", "cam2world_gl"}.
* ``Position`` is (H,W,4) float32 in the OpenGL camera frame (metres, -Z
  forward); channel 3 is z-buffer depth. World points come from
  ``p[...,3] = p[...,2] < 0`` then ``p.reshape(-1,4) @ cam2world_gl.T``
  (mani_skill2_real2sim/utils/wrappers/observation.py:142-150). That reference
  implementation mutates the array in place, so we copy first.
* ``Segmentation`` is (H,W,4) uint32; ``[...,0]`` is mesh-level and ``[...,1]``
  is actor-level (envs/custom_scenes/base_env.py:376). Links subclass Actor, so
  actor ids and link ids share one id space.
* Control mode ``arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`` with
  ``normalize_action=False``: actions reach the controller as raw metres/radians.
  ``frame="ee_align2"`` makes the xyz delta additive in the ROBOT BASE frame,
  and ``use_target=True`` accumulates deltas on the commanded target pose, so an
  open-loop cumsum is the controller's own bookkeeping rather than an
  approximation (agents/controllers/pd_ee_pose.py:103-111, 192-212).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from vls.utils.logging_utils import SteerLogger

from .base_adapter import (
    BaseEnvAdapter,
    CameraParams,
    InteractableObject,
    Pose3D,
    TrackedObject,
)

log = SteerLogger("SimplerAdapter")

# Actors that are scenery rather than manipulable objects. Mirrors the exclusion
# list SIMPLER itself uses when building the greenscreen overlay mask
# (envs/custom_scenes/base_env.py:355).
_NON_OBJECT_ACTOR_NAMES = frozenset({"ground", "goal_site", "", "arena"})

# robot_uid substring expected per embodiment, and the matching default VLM
# camera (simpler_env/utils/env/observation_utils.py:4-9).
_ROBOT_UID_SUBSTR = {"widowx": "widowx", "google_robot": "google_robot"}
_DEFAULT_CAMERA = {"widowx": "3rd_view_camera", "google_robot": "overhead_camera"}

# Checkpoint whose baked unnormalize_outputs stats fetch_policy_action_stats()
# reads for the no-policy (verify script) fallback.
GOOGLE_ROBOT_DEFAULT_CHECKPOINT = "HaomingSong/lerobot-pi0-fractal"


def fetch_policy_action_stats(checkpoint: str) -> Dict[str, np.ndarray]:
    """Read a LeRobot policy's baked action mean/std off its checkpoint file,
    without instantiating the model. Stand-in for a live ``policy`` in
    ``_probe_action_affine``'s fallback path (e.g. verify scripts)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(checkpoint, "model.safetensors")
    with safe_open(path, framework="pt") as f:
        mean = f.get_tensor("unnormalize_outputs.buffer_action.mean").numpy()
        std = f.get_tensor("unnormalize_outputs.buffer_action.std").numpy()
    return {"mean": mean, "std": std}


def _probe_action_affine(action_stats=None, embodiment="widowx", policy=None):
    """Measure the rollout's own normalized-action -> metres map for xyz.

    Calls `convert_maniskill_with_bridge_adapter` (the exact function
    run_simpler_eval_with_openpi.py applies before env.step) at three probe
    points and recovers the affine coefficients:

        f(a) = a * scale + bias      =>   bias  = f(0)
                                          scale = (f(+1) - f(-1)) / 2

    Verified affine by checking f(0) against the midpoint of f(+1), f(-1). This
    keeps the differentiable decoder bound to the real execution path instead of
    duplicating its arithmetic. Falls back to `action_stats` only if the rollout
    helper cannot be imported (e.g. running the adapter standalone).

    `policy`, when given, is applied (via its real, no_grad `unnormalize_outputs`)
    before `_conv`, so the probe walks the full x_t -> metres chain in rollout
    order. This is a no-op for WidowX (IDENTITY mapping) and the real MEAN_STD
    denormalize for Google Robot -- see the module docstring.
    """
    try:
        from eval_utils import convert_maniskill_with_bridge_adapter as _conv

        def f(v):
            a = np.zeros(7, dtype=np.float32)
            a[:3] = v
            a[6] = -1.0                      # gripper open; untouched by xyz
            if policy is not None:
                device = next(policy.parameters()).device
                with torch.no_grad():
                    a_t = torch.as_tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
                    a = policy.unnormalize_outputs({"action": a_t})["action"][0].cpu().numpy()
            return np.asarray(_conv(a, verifier_action=False,
                                    embodiment=embodiment), dtype=np.float64)[:3]

        f0, fp, fm = f(0.0), f(1.0), f(-1.0)
        scale = (fp - fm) / 2.0
        bias = f0
        if not np.allclose(f0, (fp + fm) / 2.0, atol=1e-6):
            raise RuntimeError(
                "Action map is not affine in xyz; the differentiable decoder "
                f"would be wrong. f(0)={f0}, midpoint={(fp + fm) / 2.0}"
            )
        return scale, bias
    except ImportError:
        if action_stats and "mean" in action_stats and "std" in action_stats:
            mean = np.asarray(action_stats["mean"], dtype=np.float64)
            std = np.asarray(action_stats["std"], dtype=np.float64)
            return std[:3], mean[:3]
        if not action_stats or "p01" not in action_stats or "p99" not in action_stats:
            raise ValueError(
                "SimplerAdapter needs either RL2's eval_utils on sys.path or "
                "action_stats={'p01':...,'p99':...} (bound) or "
                "{'mean':...,'std':...} (mean_std)."
            )
        p01 = np.asarray(action_stats["p01"], dtype=np.float64)
        p99 = np.asarray(action_stats["p99"], dtype=np.float64)
        return (p99[:3] - p01[:3]) / 2.0, (p99[:3] + p01[:3]) / 2.0


class SimplerAdapter(BaseEnvAdapter):
    """BaseEnvAdapter implementation for SIMPLER / ManiSkill2_real2sim."""

    def __init__(
        self,
        env,
        env_config: dict,
        device: str = "cuda",
        action_stats: Optional[Dict[str, np.ndarray]] = None,
        embodiment: Optional[str] = None,
        policy=None,
    ):
        super().__init__(env, env_config, device)

        embodiment = embodiment or env_config.get("embodiment", "widowx")
        if embodiment not in _ROBOT_UID_SUBSTR:
            raise ValueError(f"Unknown embodiment {embodiment!r}. Supported: "
                              f"{sorted(_ROBOT_UID_SUBSTR)}.")
        self.embodiment = embodiment

        uid = getattr(self.unwrapped_env, "robot_uid", "")
        if _ROBOT_UID_SUBSTR[embodiment] not in uid:
            raise NotImplementedError(
                f"SimplerAdapter configured for embodiment={embodiment!r} but "
                f"got robot_uid={uid!r}."
            )

        self.vlm_camera = env_config.get("vlm_camera") or _DEFAULT_CAMERA[embodiment]

        # Translation constants for the differentiable decoder.
        #
        # The rollout converts a normalized action to metres with
        # `convert_maniskill_with_bridge_adapter` -> INT-ACT's `postprocess` ->
        # `denormalize_bound`, a per-dimension affine map:
        #     metres = normalized * (p99-p01)/2 + (p99+p01)/2
        # We cannot call that directly here because it is numpy and guidance
        # needs autograd through this function. So we MEASURE its linearization
        # from the live adapter rather than re-deriving it from a stats file:
        # probing the real function keeps the decoder bound to whatever the
        # rollout actually executes, even if the checkpoint's statistics change.
        # `_probe_action_affine` asserts the map really is affine before use.
        #
        # The bias is NOT negligible: z is ~+7 mm/step, i.e. ~2.8 cm over a
        # 4-step chunk, so a scale-only decoder drifts systematically upward.
        scale, bias = _probe_action_affine(action_stats, embodiment=embodiment, policy=policy)
        self._act_scale = torch.as_tensor(scale, dtype=torch.float32)
        self._act_bias = torch.as_tensor(bias, dtype=torch.float32)

        self._last_obs: Optional[dict] = None
        self._last_info: dict = {}
        self._interactables: List[InteractableObject] = []
        self._seg_index_to_entity: Dict[int, Any] = {}
        self._base_rot = np.eye(3, dtype=np.float32)

    # ==================== Internals ====================

    @property
    def unwrapped_env(self):
        return getattr(self._env, "unwrapped", self._env)

    def set_obs(self, obs: dict) -> None:
        """Cache the latest observation. ManiSkill returns obs from step()/reset();
        there is no cheap way to re-render one on demand."""
        self._last_obs = obs

    def on_reset(self, obs: dict) -> None:
        """Refresh all per-episode caches. Must be called after every env.reset():
        SAPIEN actor ids are reassigned on reset, so a stale segment->entity map
        would silently attach keypoints to the wrong objects."""
        self.set_obs(obs)
        self.episode_step = 0
        self._interactables = []
        self._seg_index_to_entity = {}
        self._build_interactables()
        self._base_rot = self.get_robot_base_pose().to_rotation_matrix().astype(np.float32)

    def _require_obs(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError(
                "No cached observation. Call adapter.on_reset(obs) after env.reset() "
                "and adapter.set_obs(obs) after each env.step()."
            )
        return self._last_obs

    def _cam_images(self) -> dict:
        return self._require_obs()["image"][self.vlm_camera]

    def _cam_params(self) -> dict:
        return self._require_obs()["camera_param"][self.vlm_camera]

    # ==================== Robot State ====================

    def get_ee_pose(self) -> Pose3D:
        """EE pose in the ROBOT BASE frame (agents/base_agent.py:174-183).
        eef_pos = [xyz(3), quat_wxyz(4), gripper_openness(1)]."""
        eef = np.asarray(self._require_obs()["agent"]["eef_pos"], dtype=np.float32)
        return Pose3D(position=eef[:3], quaternion=eef[3:7])

    def get_ee_pose_world(self) -> Pose3D:
        tcp = self.unwrapped_env.tcp
        return Pose3D(position=np.asarray(tcp.pose.p), quaternion=np.asarray(tcp.pose.q))

    def get_robot_base_pose(self) -> Pose3D:
        pose = self.unwrapped_env.agent.robot.pose
        return Pose3D(position=np.asarray(pose.p), quaternion=np.asarray(pose.q))

    def get_joint_positions(self) -> np.ndarray:
        return np.asarray(self._require_obs()["agent"]["qpos"], dtype=np.float32)

    def get_gripper_state(self) -> float:
        return float(self._require_obs()["agent"]["eef_pos"][7])

    # ==================== Actions ====================

    def delta_actions_to_ee_trajectory(
        self,
        action_sequence: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Decode a NORMALIZED action chunk into a 3D world-frame EE trajectory.

        Differentiable end-to-end: VLS backprops the guidance reward through this
        to get d(reward)/d(noisy action). Uses only out-of-place ops so the graph
        survives.

        Chain (see module docstring):
            normalized -> denormalize_bound (per-dim scale + bias) -> metres in
            BASE frame -> rotate to WORLD -> cumsum onto current EE position.

        Args:
            action_sequence: (T, >=3) normalized actions. Only xyz is read; the
                gripper channel is deliberately never denormalized.

        Returns:
            (T+1, 3) world-frame positions, starting at the current EE position.
        """
        a = action_sequence
        if not torch.is_tensor(a):
            a = torch.as_tensor(a, dtype=torch.float32)
        device, dtype = a.device, a.dtype

        scale = self._act_scale.to(device=device, dtype=dtype)
        bias = self._act_bias.to(device=device, dtype=dtype)

        delta_base = a[:, :3] * scale + bias                      # (T,3) metres, BASE
        rot = torch.as_tensor(self._base_rot, device=device, dtype=dtype)
        delta_world = delta_base @ rot.T                          # (T,3) WORLD

        cum = torch.cumsum(delta_world, dim=0)
        start = torch.as_tensor(
            self.get_ee_pose_world().position, device=device, dtype=dtype
        )
        return torch.cat([start.unsqueeze(0), start + cum], dim=0)  # (T+1,3)

    def get_action_space_info(self) -> Dict[str, Any]:
        return {
            "dim": 7,
            "type": "delta_ee_pose",
            "normalized": True,
            "frame": "base",
            "pos_scale": self._act_scale.cpu().numpy(),
            "pos_bias": self._act_bias.cpu().numpy(),
        }

    # ==================== Camera & Perception ====================

    def get_camera_names(self) -> List[str]:
        obs = self._require_obs()
        return list(obs.get("image", {}).keys())

    def get_camera_params(self, camera_name: str = None) -> CameraParams:
        """Intrinsics/extrinsics in OpenCV convention, which is what the inherited
        BaseEnvAdapter.project_3d_to_2d expects."""
        name = camera_name or self.vlm_camera
        params = self._require_obs()["camera_param"][name]
        color = self._require_obs()["image"][name]["Color"]
        h, w = color.shape[:2]
        return CameraParams(
            intrinsic=np.asarray(params["intrinsic_cv"], dtype=np.float32),
            extrinsic=np.asarray(params["extrinsic_cv"], dtype=np.float32),
            width=w,
            height=h,
        )

    def get_vlm_image(self) -> np.ndarray:
        """RGB uint8 (H,W,3) for the VLM and for visualization.

        This is the greenscreened composite: SIMPLER replaces BACKGROUND pixels
        with a real photo while keeping the robot and target objects
        sim-rendered (base_env.py:390-393). So the image is photorealistic even
        though background *geometry* in the point cloud is still simulated. That
        is harmless here because background pixels carry segment index 0 and are
        excluded from keypoint detection.

        No vertical flip is applied: SAPIEN renders in top-down row order like
        OpenCV. (LIBERO/MuJoCo needed flipud; SIMPLER does not. V1a's
        reprojection round-trip is what proves this.)
        """
        color = self._cam_images()["Color"]
        return np.clip(np.asarray(color)[..., :3] * 255.0, 0, 255).astype(np.uint8)

    def _depth_and_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Unproject the Position texture into a world-frame point cloud."""
        # Copy first: the reference implementation writes into this array.
        pos = np.array(self._cam_images()["Position"], dtype=np.float64, copy=True)
        cam2world = np.asarray(self._cam_params()["cam2world_gl"], dtype=np.float64)

        h, w = pos.shape[:2]
        valid = pos[..., 2] < 0                      # GL looks down -Z
        depth = (-pos[..., 2]).astype(np.float32)    # metres, positive forward

        pos[..., 3] = valid                          # w=1 hit, w=0 background
        xyzw = pos.reshape(-1, 4) @ cam2world.T
        points = xyzw[:, :3].reshape(h, w, 3).astype(np.float32)
        points[~valid] = 0.0                         # zero out misses

        return depth, points

    def _build_interactables(self) -> List[InteractableObject]:
        """Enumerate manipulable actors + articulated links, assigning segment
        indices from 1 (0 is reserved for background).

        Follows base_env.py:355-370. Robot links are deliberately EXCLUDED: a
        keypoint attached to the gripper would move every step and poison the
        guidance reward.
        """
        env = self.unwrapped_env
        robot_link_ids = {link.id for link in env.agent.robot.get_links()}

        entities = [
            a for a in env.get_actors()
            if a.name not in _NON_OBJECT_ACTOR_NAMES and a.id not in robot_link_ids
        ]
        for art in env._scene.get_all_articulations():
            if art is env.agent.robot:
                continue
            entities.extend(art.get_links())

        interactables: List[InteractableObject] = []
        seg_map: Dict[int, Any] = {}
        for seg_idx, entity in enumerate(entities, start=1):
            interactables.append(
                InteractableObject(
                    name=entity.name,
                    object_id=int(entity.id),
                    link_index=-1,
                    segment_index=seg_idx,
                    obj_ref=entity,
                )
            )
            seg_map[seg_idx] = entity

        self._interactables = interactables
        self._seg_index_to_entity = seg_map
        return interactables

    def get_interactable_objects(self) -> List[InteractableObject]:
        if not self._interactables:
            self._build_interactables()
        return self._interactables

    def process_segmentation(
        self,
        seg_image: np.ndarray,
    ) -> Tuple[np.ndarray, List[InteractableObject], Dict[int, str]]:
        """Map SAPIEN's actor-level segmentation onto contiguous segment indices.

        Args:
            seg_image: (H,W,4) uint32 Segmentation texture.

        Returns:
            processed_seg: (H,W) int32, 0 = background.
            interactables, segment_id_to_name
        """
        seg = np.asarray(seg_image)
        actor_seg = seg[..., 1] if seg.ndim == 3 else seg   # [...,1] = actor level

        interactables = self.get_interactable_objects()
        processed = np.zeros(actor_seg.shape, dtype=np.int32)
        segment_id_to_name: Dict[int, str] = {0: "background"}

        for obj in interactables:
            mask = actor_seg == obj.object_id
            if mask.any():
                processed[mask] = obj.segment_index
                segment_id_to_name[obj.segment_index] = obj.name

        return processed, interactables, segment_id_to_name

    def get_keypoint_detection_inputs(
        self,
        camera_name: str = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
        """Everything KeypointDetector.get_keypoints() needs.

        Returns:
            rgb (H,W,3) uint8, depth (H,W) float32 metres,
            points (H,W,3) float32 WORLD frame, segmentation (H,W) int32,
            segment_id_to_name
        """
        if camera_name is not None and camera_name != self.vlm_camera:
            raise ValueError(
                f"SimplerAdapter is configured for camera {self.vlm_camera!r}, got {camera_name!r}"
            )

        rgb = self.get_vlm_image()
        depth, points = self._depth_and_points()
        segmentation, _, segment_id_to_name = self.process_segmentation(
            self._cam_images()["Segmentation"]
        )
        return rgb, depth, points, segmentation, segment_id_to_name

    # ==================== Scene Objects ====================

    def get_scene_objects(self, exclude_names: Optional[List[str]] = None) -> List[TrackedObject]:
        exclude = set(exclude_names or [])
        objects = []
        for obj in self.get_interactable_objects():
            if obj.name in exclude:
                continue
            entity = obj.obj_ref
            objects.append(
                TrackedObject(
                    name=obj.name,
                    pose=Pose3D(
                        position=np.asarray(entity.pose.p),
                        quaternion=np.asarray(entity.pose.q),
                    ),
                    obj_ref={"segment_index": obj.segment_index, "entity": entity},
                )
            )
        return objects

    def get_object_pose(self, object_name: str) -> Optional[Pose3D]:
        for obj in self.get_interactable_objects():
            if obj.name == object_name:
                e = obj.obj_ref
                return Pose3D(position=np.asarray(e.pose.p), quaternion=np.asarray(e.pose.q))
        return None

    def get_object_pose_by_segment(self, segment_index: int) -> Optional[Pose3D]:
        """Live world pose for a segment index. This is what KeypointTracker uses
        each chunk to re-derive keypoint positions as objects move."""
        entity = self._seg_index_to_entity.get(int(segment_index))
        if entity is None:
            return None
        return Pose3D(position=np.asarray(entity.pose.p), quaternion=np.asarray(entity.pose.q))

    # ==================== Task / Episode ====================

    def get_instruction(self) -> str:
        return self.unwrapped_env.get_language_instruction()

    def get_task_description(self) -> str:
        return self.get_instruction()

    def get_task_info(self) -> Dict[str, Any]:
        """Per-task metadata. Upstream VLS reads recommended_guide_scale each
        episode and overrides its guide scale with it (main.py:472-476)."""
        scales = self._env_config.get("task_guide_scales", [])
        task_id = self._env_config.get("task_id", 0)
        recommended = None
        if isinstance(scales, (list, tuple)) and task_id < len(scales):
            recommended = scales[task_id]
        if recommended is None:
            recommended = self._env_config.get("guide_scale", 80.0)

        return {
            "instruction": self.get_instruction(),
            "recommended_guide_scale": recommended,
            "task_type": "manipulation",
            "requires_precision": True,
            "task_id": task_id,
        }

    def check_success(self) -> Tuple[bool, str]:
        return bool(self._last_info.get("success", False)), ""

    def get_behavior_static(self) -> Dict[str, int]:
        # CALVIN-specific bookkeeping upstream; unused on SIMPLER but required
        # by the abstract contract.
        return {}

    def get_policy_observation(self, sample_num: int = 1) -> Dict[str, Any]:
        # RL2 builds policy observations through its own preprocessing adapter
        # (run_simpler_eval_with_openpi.py:383), so duplicating that here would
        # create a second path that can silently drift out of sync.
        raise NotImplementedError(
            "SIMPLER policy observations are built by RL2's preprocess_adapter; "
            "this adapter is used for perception, geometry and action decoding only."
        )

    # ==================== Gym passthrough ====================

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.set_obs(obs)
        self._last_info = info or {}
        self.episode_step += 1
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        obs, info = self._env.reset(**kwargs)
        self._last_info = info or {}
        self.on_reset(obs)
        return obs, info

    def get_obs(self) -> Dict:
        return self._last_obs if self._last_obs is not None else {}
