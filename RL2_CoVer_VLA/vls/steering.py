"""
VLS steering controller: per-episode setup and the closed-loop stage machine.

Bundles the pieces that upstream VLS spreads across ``main.py`` — keypoint
detection/registration, guidance-function synthesis or loading, and stage
recognition — behind a small interface the RL2 rollout can call at two points:

    steer.begin_episode(...)   once, after the scene has settled
    steer.on_chunk(...)        at each chunk boundary, before sampling

Stage recognition (ported from ``main.py:_update_stage``, lines 384-453) is a
Schmitt trigger on the normalized guidance reward plus gripper open/close
transitions. When a trigger fires it optionally asks a VLM which stage the
episode is in; the VLM call is rate-limited, and the whole mechanism degrades
gracefully to "stay in the current stage" when no VLM is configured.

Two upstream behaviors deliberately preserved:

* **Per-episode guidance regeneration.** Object placement is randomized on
  reset, so keypoints are re-detected and guidance re-synthesized each episode
  (``main.py:482``). Pass ``guidance_dir`` to replay cached functions instead
  and skip the API call entirely.
* **Per-episode error isolation.** Guidance functions are LLM-written code
  executed via ``exec``; one bad function must not kill a long benchmark run
  (``main.py:483-516``). Failures here disable guidance for the episode rather
  than propagating.
"""

import json
import os
import traceback
from typing import Callable, Dict, List, Optional

import numpy as np

from vls.utils.guidance_utils import load_functions_from_txt
from vls.utils.logging_utils import SteerLogger

log = SteerLogger("VLSSteering")


class VLSSteeringController:
    """Owns per-episode VLS state: keypoints, guidance functions, stage."""

    def __init__(
        self,
        adapter,
        detector,
        tracker,
        sampler,
        guidance_dir: Optional[str] = None,
        vlm_agent=None,
        stage_recognizer=None,
        vlm_query_limit: int = 10,
        schmitt_upper: float = 0.8,
        schmitt_lower: float = 0.6,
        default_output_dir: str = "outputs/vls",
        task_key: Optional[str] = None,
    ):
        self.adapter = adapter
        self.detector = detector
        self.tracker = tracker
        self.sampler = sampler
        self.guidance_dir = guidance_dir
        self.vlm_agent = vlm_agent
        self.stage_recognizer = stage_recognizer
        self.vlm_query_limit = vlm_query_limit
        self.schmitt_upper = schmitt_upper
        self.schmitt_lower = schmitt_lower
        self.default_output_dir = default_output_dir
        # Identifies which task's guidance a cache directory holds. Keypoint
        # indices are only meaningful for the task they were generated on.
        self.task_key = task_key

        self.guidance_fns: Dict[int, List[Callable]] = {}
        self.stage_descriptions = ""
        self.keypoint_id_to_object: Dict[int, str] = {}
        self.init_img_with_keypoints = None
        self.reset_episode()

    # -- state ------------------------------------------------------------- #

    def reset_episode(self):
        self.ready = False              # True once keypoints + guidance are loaded
        self.current_stage = 1
        self.use_guidance = True
        self.keypoints = None
        self._prev_norm_reward = 0.0
        self._prev_gripper_open = None
        self._vlm_query_count = 0
        self.guidance_fns = {}
        if self.tracker is not None:
            self.tracker.reset()
        if self.sampler is not None:
            self.sampler.reset_episode()

    # -- episode setup ----------------------------------------------------- #

    def begin_episode(self, instruction: str, episode_dir: Optional[str] = None) -> bool:
        """Detect keypoints and load/synthesize guidance for this episode.

        Call once per episode AFTER the scene has settled: objects are still
        falling during the warm-up steps, and keypoints registered then would
        attach to mid-fall poses.

        Returns True if steering is ready; False disables guidance for the
        episode (the rollout continues unguided rather than aborting).
        """
        try:
            rgb, _, points, seg, seg_names = self.adapter.get_keypoint_detection_inputs()

            keypoints, projected_img, mask_ids = self.detector.get_keypoints(
                rgb, points, seg, seg_names
            )
            if keypoints is None or len(keypoints) == 0:
                log.warning("No keypoints detected; running this episode unguided")
                return False

            kp_to_object = self.tracker.register_keypoints(keypoints, mask_ids, seg_names)
            self.keypoint_id_to_object = kp_to_object
            self.init_img_with_keypoints = projected_img
            log.info(f"Registered {len(keypoints)} keypoints: "
                     + ", ".join(f"{i}:{n}" for i, n in sorted(kp_to_object.items())))

            guidance_dir = self._resolve_guidance(
                instruction, keypoints, kp_to_object, projected_img, episode_dir
            )
            if guidance_dir is None:
                return False

            self._load_guidance(guidance_dir)
            if not any(self.guidance_fns.values()):
                log.warning("No guidance functions loaded; running this episode unguided")
                return False

            self.keypoints = self.tracker.get_keypoint_positions()
            self.current_stage = 1
            self.use_guidance = True
            self.ready = True
            return True

        except Exception as e:
            # LLM-written guidance can raise anything; isolate to this episode.
            log.error(f"Episode steering setup failed: {e}")
            log.debug(traceback.format_exc())
            if episode_dir:
                try:
                    os.makedirs(episode_dir, exist_ok=True)
                    with open(os.path.join(episode_dir, "error.txt"), "w") as f:
                        f.write(f"VLS setup failed:\n{traceback.format_exc()}")
                except OSError:
                    pass
            return False

    def _resolve_guidance(self, instruction, keypoints, kp_to_object,
                          projected_img, episode_dir) -> Optional[str]:
        """Cached guidance directory, or synthesize a fresh one via the VLM.

        Upstream VLS has no task lookup here: `cached_functions_dir` is a single
        path (main.py:343-347). That is safe there only because `run_main.sh`
        invokes one task suite per run, so a given directory holds exactly one
        task's guidance. Our eval script loops over several suites in one run,
        so the same convention has to be made explicit: we look for a
        per-task subdirectory under `guidance_dir` and fall back to the bare
        path. Reusing one task's guidance on another would silently target
        keypoint indices belonging to different objects.
        """
        if self.guidance_dir:
            candidates = []
            if self.task_key:
                candidates.append(os.path.join(self.guidance_dir, self.task_key))
            candidates.append(self.guidance_dir)

            for path in candidates:
                if os.path.exists(os.path.join(path, "metadata.json")):
                    log.info(f"Using cached guidance functions from {path}")
                    return path

            log.error(
                f"No cached guidance (metadata.json) found under "
                f"{' or '.join(candidates)}; run once with an empty "
                f"vls_guidance_dir to populate it."
            )
            return None

        if self.vlm_agent is None:
            log.error("No guidance_dir and no VLM agent configured")
            return None

        # VLMAgent.generate_guidance() writes query_img.png / prompt.txt /
        # output_raw.txt / metadata.json / stage*_guidance.txt into task_dir, and
        # has no default for it, so always set one.
        #
        # If the caller asked for caching but the cache was empty, write the
        # freshly generated set into the task-scoped cache slot so a later run
        # with the same vls_guidance_dir replays it offline.
        if self.guidance_dir and self.task_key:
            task_dir = os.path.join(self.guidance_dir, self.task_key)
        else:
            task_dir = os.path.join(episode_dir or self.default_output_dir, "vlm_agent")
        self.vlm_agent.task_dir = task_dir
        os.makedirs(task_dir, exist_ok=True)

        metadata = {
            "init_keypoint_positions": keypoints,
            "num_keypoints": len(keypoints),
            "key_points_objects_map": kp_to_object,
        }
        return self.vlm_agent.generate_guidance(projected_img, instruction, metadata)

    def _load_guidance(self, guidance_dir: str):
        """Load per-stage guidance functions. num_stages comes from metadata.json."""
        meta_path = os.path.join(guidance_dir, "metadata.json")
        with open(meta_path) as f:
            program_info = json.load(f)

        num_stages = int(program_info.get("num_stages", 1))
        self.guidance_fns = {}
        for stage in range(1, num_stages + 1):
            path = os.path.join(guidance_dir, f"stage{stage}_guidance.txt")
            self.guidance_fns[stage] = load_functions_from_txt(path) if os.path.exists(path) else []
            log.info(f"Stage {stage}: {len(self.guidance_fns[stage])} guidance functions loaded")
        log.info(f"Loaded {len(self.guidance_fns)} stages total")

        if self.vlm_agent is not None:
            raw = os.path.join(guidance_dir, "output_raw.txt")
            if os.path.exists(raw):
                self.stage_descriptions = self.vlm_agent._extract_stage_descriptions_from_output(raw)

    # -- per-chunk update -------------------------------------------------- #

    def on_chunk(self, gripper_val: Optional[float] = None):
        """Refresh keypoints and update the stage. Call at each chunk boundary.

        Matches upstream's cadence: keypoints and stage are updated only when a
        new chunk is generated (``main.py:573-590``), not on every env step.

        Returns:
            (keypoints, guidance_fns_for_current_stage) — the latter is None
            when guidance is currently disabled.
        """
        if not self.ready:
            return None, None

        self.keypoints = self.tracker.get_keypoint_positions()
        self._update_stage(gripper_val)

        fns = self.guidance_fns.get(self.current_stage, []) if self.use_guidance else None
        if self.use_guidance and not fns:
            log.warning(f"No guidance functions for stage {self.current_stage}, disabling")
            self.use_guidance = False
            fns = None

        return self.keypoints, fns

    def _update_stage(self, gripper_val: Optional[float]):
        """Schmitt trigger on reward + gripper transitions, gating a VLM query."""
        curr_reward = self.sampler.get_normalized_reward()

        # BridgeSimplerAdapter.postprocess_gripper maps model gripper output
        # (0=close,1=open) to -1=CLOSE, +1=OPEN for the simpler/ManiSkill env
        # (INT-ACT/src/experiments/env_adapters/simpler.py:218-224). So on the
        # EXECUTED action channel this reads: action < 0 = CLOSE, action > 0 = OPEN.
        # This was previously inverted, which fed the Schmitt trigger below (and
        # the "gripper opened"/"gripper closed" reason strings sent to the stage
        # VLM) the wrong event on every real open/close transition.
        curr_open = (gripper_val > 0) if gripper_val is not None else None
        changed = (self._prev_gripper_open is not None and curr_open is not None
                   and curr_open != self._prev_gripper_open)

        trigger = None
        if changed and not curr_open:
            trigger = "gripper closed"
        elif changed and curr_open:
            trigger = "gripper opened"
        elif self.use_guidance and self._prev_norm_reward < self.schmitt_upper <= curr_reward:
            trigger = f"reward rose above {self.schmitt_upper:.0%}"
        elif self.use_guidance and self._prev_norm_reward > self.schmitt_lower >= curr_reward:
            trigger = f"reward dropped below {self.schmitt_lower:.0%}"

        if trigger and self.stage_recognizer is not None:
            if self._vlm_query_count < self.vlm_query_limit:
                log.info(f"[Trigger] {trigger} "
                         f"(query {self._vlm_query_count + 1}/{self.vlm_query_limit})")
                try:
                    new_stage, need_guidance = self.stage_recognizer.identify_stage_and_guidance(
                        current_rgb=np.array(self.adapter.get_vlm_image()),
                        instruction=self.adapter.get_task_description(),
                        stage_descriptions=self.stage_descriptions,
                        init_img_with_keypoints=self.init_img_with_keypoints,
                        keypoint_id_to_object=self.keypoint_id_to_object,
                        num_stages=len(self.guidance_fns),
                        trigger_reason=trigger,
                    )
                    self._vlm_query_count += 1

                    if new_stage != self.current_stage or need_guidance != self.use_guidance:
                        log.info(f"[VLM] Stage: {self.current_stage} -> {new_stage}, "
                                 f"Guidance: {self.use_guidance} -> {need_guidance}")
                        if new_stage != self.current_stage:
                            self.sampler.reset_stage()   # clear the reward baseline
                            curr_reward = 0.0
                    self.current_stage = new_stage
                    self.use_guidance = need_guidance
                except Exception as e:
                    log.warning(f"Stage recognition failed, keeping stage "
                                f"{self.current_stage}: {e}")
            else:
                log.debug(f"[Trigger] {trigger} (skipped, limit reached)")

        self._prev_norm_reward = curr_reward
        self._prev_gripper_open = curr_open
