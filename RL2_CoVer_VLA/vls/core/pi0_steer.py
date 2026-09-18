"""
VLS guided sampling for RL2-VLA's pi0 flow-matching policy.

Adapted from upstream VLS ``core/pi05_steer.py`` (``_sample_actions_guided``,
``_compute_keypoint_gradient``, ``_compute_diversity_gradient``) and wired to
match RL2's existing ``rl2_utils.get_composed_actions`` loop, so it can be
swapped in behind a config flag with an identical return signature.

Three steering mechanisms, split by phase along the flow-matching time axis:

1. ``time > start_time``  — RBF diversity: pairwise inverse-distance repulsion
   between the decoded EE trajectories, ADDED to the velocity so particles
   spread out. Self-disables at batch size 1 (needs >= 2 particles).
2. ``time <= start_time`` — keypoint gradient guidance: decode the noisy sample
   to a 3D trajectory, score it with the VLM-written reward, backprop, and
   SUBTRACT the unit-normalized gradient from the velocity's xyz channels. The
   scale decays via a sigmoid in the normalized reward, so guidance fades out
   as the stage's goal is approached ("graceful hand-off").
3. FK resampling — SMC particle filter over the same reward; resamples and,
   at the terminal step, sorts particles so index 0 is the best one.

Deviations from upstream, all deliberate (see the port plan):

* ``denoise_step`` here returns ``(v_t, embeds)``; upstream pi05 returns just
  ``v_t``. Every call site unpacks.
* The Euler update is out-of-place (``x = x + dt*v``); upstream pi0's
  ``sample_actions`` uses ``x_t += dt * v_t``, which would break autograd.
* ``start_time`` defaults correctly. Upstream ``pi05_steer.py:191`` reads
  ``start_ratio if start_ratio is None else 0.8``, which is inverted and
  silently ignores an explicit ratio.
* FKD is wired with integer step indices, as ``diffusion_policy_steer.py:330``
  does — NOT ``pi05_steer.py``'s ``linspace(1.0, 0.0, ...)``, which
  ``.long()``-truncates to ``[1,0,0,...]`` so ``resample()`` early-returns on
  every step and silently disables FK steering.
"""

from collections import deque
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch

from vls.core.fkd_class import FKD
from vls.utils.logging_utils import SteerLogger

log = SteerLogger("PI0Steer")

GuidanceFns = Union[Callable, Sequence[Callable], None]


# --------------------------------------------------------------------------- #
# Reward / gradient helpers
# --------------------------------------------------------------------------- #

def _eval_guidance(guidance_fn: GuidanceFns, keypoints: torch.Tensor,
                   traj: torch.Tensor):
    """Evaluate one or many guidance functions on a (B,T,3) trajectory."""
    if guidance_fn is None:
        return None
    if isinstance(guidance_fn, (list, tuple)):
        if not guidance_fn:
            return None
        return sum(fn(keypoints, traj) for fn in guidance_fn)
    return guidance_fn(keypoints, traj)


def compute_keypoint_gradient(
    sample: torch.Tensor,
    keypoints: torch.Tensor,
    guidance_fn: GuidanceFns,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    horizon: int,
) -> tuple:
    """d(reward)/d(noisy sample) for the VLM guidance reward.

    Differentiates only through ``decode_fn`` (the trajectory decoder), never
    through the policy — so the numpy/jax round-trips elsewhere in the loop are
    harmless, and the graph is rebuilt from a detached sample each call.

    Because the decoder is affine in the action (scale + bias, then a constant
    rotation and cumsum), the per-dimension Jacobian is applied automatically;
    there is nothing extra to chain here.

    Returns:
        (unit-normalized gradient or None, raw reward value)
    """
    if not guidance_fn:
        return None, 0.0

    try:
        with torch.enable_grad():
            grad_sample = sample.detach().requires_grad_(True)
            traj = decode_fn(grad_sample)[:, :horizon, :3]

            reward = _eval_guidance(guidance_fn, keypoints, traj)
            if reward is None:
                return None, 0.0
            if isinstance(reward, (int, float)):
                return None, float(reward)
            if not getattr(reward, "requires_grad", False):
                return None, float(reward.item() if hasattr(reward, "item") else reward)

            if reward.dim() > 0:
                reward = reward.sum()
            reward_value = float(reward.item())

            grad = torch.autograd.grad(reward, grad_sample)[0]
            norm = torch.norm(grad)
            if norm > 1e-8:
                grad = grad / (norm + 1e-8)
            return grad, reward_value
    except Exception as e:  # VLM-written code can raise; degrade, don't crash
        log.warning(f"Guidance gradient failed: {e}")
        return None, 0.0


def compute_diversity_gradient(
    sample: torch.Tensor,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    horizon: int,
) -> Optional[torch.Tensor]:
    """RBF (inverse-distance) repulsion gradient between particle trajectories."""
    batch = sample.shape[0]
    if batch < 2:
        return None

    try:
        with torch.enable_grad():
            grad_sample = sample.detach().requires_grad_(True)
            traj = decode_fn(grad_sample)[:, :horizon, :3]
            flat = traj.reshape(batch, -1)

            diff = flat.unsqueeze(1) - flat.unsqueeze(0)
            dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-6)
            mask = ~torch.eye(batch, dtype=torch.bool, device=sample.device)
            potential = ((1.0 / (dist + 1e-6)) * mask.float()).sum()

            grad = torch.autograd.grad(potential, grad_sample)[0]
            norm = torch.norm(grad)
            if norm > 1e-8:
                grad = grad / (norm + 1e-8)
            return grad
    except Exception as e:
        log.warning(f"Diversity gradient failed: {e}")
        return None


# --------------------------------------------------------------------------- #
# Guided sampler
# --------------------------------------------------------------------------- #

class GuidedSampler:
    """Runs VLS-steered flow-matching sampling for the pi0 policy.

    Holds the per-stage reward baseline that the adaptive guidance scale needs,
    so it must persist across chunks within an episode (``reset_stage()`` on a
    stage change, ``reset_episode()`` between episodes).
    """

    def __init__(self, pi0_policy, adapter, cfg):
        self.policy = pi0_policy
        self.adapter = adapter
        self.cfg = cfg
        self.reset_episode()

    # -- state ------------------------------------------------------------- #

    def reset_episode(self):
        self._stage_init_reward = None
        self._last_normalized_reward = 0.0
        self._last_scale = 0.0
        # {"pi0": (3,), "vls": (3,)} world-frame displacements of the two terms
        # summed to form the denoising velocity. None until guidance first runs.
        self._last_terms = None

    def reset_stage(self):
        """Clear the per-stage reward baseline (call on stage transitions)."""
        self._stage_init_reward = None

    def get_normalized_reward(self) -> float:
        return self._last_normalized_reward

    def get_last_scale(self) -> float:
        return self._last_scale

    def get_last_terms(self):
        """World-frame displacements of the two summed velocity terms, or None.

        {"pi0": (3,), "vls": (3,)} -- these add to the net denoising step, so
        their relative magnitudes are meaningful and must be drawn to a common
        scale.
        """
        return self._last_terms

    # -- decoding ---------------------------------------------------------- #

    def _term_to_world(self, term: torch.Tensor) -> np.ndarray:
        """One velocity term -> the world-frame displacement it contributes.

        `term` is (T,3) in normalized action units. Applying the decoder's own
        linear part (per-axis scale, then base->world rotation) and summing over
        the chunk gives the displacement this term adds to the chunk endpoint.

        The affine bias is deliberately NOT applied: it is a constant offset of
        the decode, not a contribution of either term, and adding it to both
        would distort their relative magnitudes.
        """
        t = term.detach().to(torch.float32).cpu()
        scale = self.adapter._act_scale.to(t.device, t.dtype)
        rot = torch.as_tensor(self.adapter._base_rot, dtype=t.dtype)
        return ((t * scale).sum(dim=0) @ rot.T).numpy()

    def _make_decode_fn(self, horizon: int) -> Callable:
        """Map a padded, normalized (B,T,D) sample to (B,T+1,3) world trajectories.

        Differentiable throughout; the per-sample loop preserves the graph
        because results are re-stacked rather than written in place.
        """
        original_dim = self.policy.config.action_feature.shape[0]

        def decode(sample: torch.Tensor) -> torch.Tensor:
            actions = sample[:, :horizon, :original_dim]
            return torch.stack(
                [self.adapter.delta_actions_to_ee_trajectory(actions[b])
                 for b in range(actions.shape[0])],
                dim=0,
            )

        return decode

    # -- main loop --------------------------------------------------------- #

    @torch.no_grad()
    def sample(
        self,
        processed_obs: dict,
        image_key: str,
        task_list: List[str],
        batch_size: int,
        keypoints: Optional[np.ndarray],
        guidance_fn: GuidanceFns,
        action_noise_std: float = 1.0,
        guide_scale: float = 80.0,
        diversity_scale: float = 20.0,
        start_ratio: Optional[float] = None,
        sigmoid_k: float = 25.0,
        sigmoid_x0: float = 0.75,
        use_diversity: bool = True,
        use_fkd: bool = False,
        fkd_config: Optional[dict] = None,
        global_step: int = 0,
        current_stage: int = 1,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Denoise one action chunk under VLS steering.

        Returns:
            (B, n_action_steps, action_dim) unnormalized actions, same format as
            ``rl2_utils.get_composed_actions``.
        """
        # Clear the overlay diagnostics for this chunk. Without this they would
        # persist from the last guided chunk, so the video would keep drawing a
        # VLS arrow on frames where the stage machine has turned guidance OFF.
        self._last_terms = None

        # One fixed prompt for the whole batch: the batch dimension is particles,
        # not a prompt mixture (upstream broadcasts a single instruction).
        assert len(set(task_list)) == 1, (
            f"VLS expects a single prompt across the batch, got {len(set(task_list))}. "
            "Set lang_rephrase_num=1."
        )

        policy, model = self.policy, self.policy.model
        horizon = policy.config.n_action_steps
        original_dim = policy.config.action_feature.shape[0]

        observation = {
            image_key: processed_obs["observation.images.top"].repeat(batch_size, 1, 1, 1),
            "observation.state": processed_obs["observation.state"].repeat(batch_size, 1),
            "task": task_list,
        }

        past_key_values, state, prefix_pad_masks = policy.forward_pass_vlm(
            observation, noise_std=action_noise_std
        )

        device = state.device
        num_steps = model.config.num_steps
        x_t = model.sample_noise(
            (batch_size, model.config.chunk_size, model.config.max_action_dim),
            device, noise_std=action_noise_std,
        )

        start_time = 0.8 if start_ratio is None else float(start_ratio)
        decode_fn = self._make_decode_fn(horizon)

        keypoints_t = None
        if keypoints is not None and len(keypoints):
            keypoints_t = torch.as_tensor(np.asarray(keypoints), device=device, dtype=torch.float32)
        use_guidance = bool(guidance_fn) and keypoints_t is not None

        fkd = self._init_fkd(
            fkd_config if use_fkd else None, batch_size, num_steps, start_time,
            keypoints_t, guidance_fn, decode_fn, horizon, device,
        )

        dt = -1.0 / num_steps
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        reward_history = []
        step_idx = 0

        while time >= -dt / 2:
            v_t, _ = model.denoise_step(              # NOTE: returns a tuple
                state, prefix_pad_masks, past_key_values, x_t, time.expand(batch_size)
            )

            # pi0's own velocity exists on EVERY denoising step, guided or not,
            # so record it unconditionally. The guided branch below adds the VLS
            # term; when guidance is off this stays the only entry and the
            # overlay draws just the pi0 arrow.
            #
            # Scaled by dt: the arrow must show the signed DISPLACEMENT this
            # term contributes to x_t (x_t = x_t + dt*v_t), not the raw
            # velocity. dt is negative, so omitting it here would draw the
            # arrow exactly backwards -- which is what made the VLS arrow
            # below appear to point away from the target object.
            self._last_terms = {"pi0": self._term_to_world(dt * v_t[0, :horizon, :3])}

            if use_diversity and time > start_time and batch_size > 1:
                div = compute_diversity_gradient(x_t, decode_fn, horizon)
                if div is not None:
                    v_t = v_t.clone()
                    v_t[:, :horizon, :3] += diversity_scale * div[:, :horizon, :3]

            elif use_guidance and time <= start_time:
                grad, reward_value = compute_keypoint_gradient(
                    x_t, keypoints_t, guidance_fn, decode_fn, horizon
                )
                if grad is not None:
                    # Progress within the stage: 0 at the baseline, ~1 at the goal.
                    if self._stage_init_reward is not None and self._stage_init_reward < -1e-6:
                        norm_r = 1.0 - (reward_value / self._stage_init_reward)
                        norm_r = float(np.clip(norm_r, 0.0, 1.2))
                    else:
                        norm_r = 0.0
                    self._last_normalized_reward = norm_r
                    reward_history.append((step_idx, reward_value, norm_r))

                    strength = 1.0 / (1.0 + np.exp(sigmoid_k * (norm_r - sigmoid_x0)))
                    scale = guide_scale * strength
                    self._last_scale = scale

                    if verbose and step_idx == int(start_time * num_steps):
                        init = f"{self._stage_init_reward:.6f}" if self._stage_init_reward is not None else "None"
                        log.info(
                            f"[Step {global_step}] Stage {current_stage} | "
                            f"reward={reward_value:.6f}, init={init}, "
                            f"norm_r={norm_r:.3f}, sig_strength={strength:.3f}, scale={scale:.2f}"
                        )

                    # Overlay diagnostics: the two terms that are SUMMED to form
                    # the denoising velocity at this step,
                    #     v = v_pi0 + (-scale * grad)
                    # kept in the same units so their magnitudes are directly
                    # comparable (the overlay draws them to a common scale).
                    # Summed over the chunk horizon and rotated base->world, so
                    # each arrow is the world-frame displacement that term
                    # contributes to the chunk endpoint -- which is dt times
                    # the velocity term (x_t = x_t + dt*v_t), NOT the raw
                    # term. dt < 0, so this also flips the sign: the actual
                    # update moves toward the target (since grad points
                    # toward it and -dt > 0 cancels the "-scale*grad" sign),
                    # while the undiluted "-scale*grad" points away from it.
                    self._last_terms["vls"] = self._term_to_world(
                        dt * (-scale * grad[0, :horizon, :3])
                    )

                    v_t = v_t.clone()
                    v_t[:, :horizon, :3] -= scale * grad[:, :horizon, :3]

            x_t = x_t + dt * v_t          # out-of-place: upstream pi0 uses +=

            if fkd is not None and time <= start_time:
                x_t, _ = fkd.resample(sampling_idx=step_idx, latents=x_t, x0_preds=x_t)
                idx = getattr(fkd, "last_indices", None)
                if idx is not None:
                    past_key_values, prefix_pad_masks, state = _reindex_particles(
                        idx, past_key_values, prefix_pad_masks, state
                    )

            time = time + dt
            step_idx += 1

        # Baseline for the sigmoid comes from the FIRST chunk's final reward.
        if reward_history and self._stage_init_reward is None:
            self._stage_init_reward = reward_history[-1][1]
            log.info(
                f"[Step {global_step}] Stage {current_stage} "
                f"init_reward={self._stage_init_reward:.6f} (from first chunk's final step)"
            )

        actions = x_t[:, :horizon, :original_dim]
        return policy.unnormalize_outputs({"action": actions})["action"]

    # -- FKD --------------------------------------------------------------- #

    def _init_fkd(self, fkd_config, batch_size, num_steps, start_time,
                  keypoints_t, guidance_fn, decode_fn, horizon, device):
        if fkd_config is None or batch_size <= 1 or keypoints_t is None or not guidance_fn:
            return None

        def reward_fn(x0_preds):
            traj = decode_fn(x0_preds)[:, 1:horizon, :3]
            rewards = []
            for b in range(traj.shape[0]):
                try:
                    r = _eval_guidance(guidance_fn, keypoints_t, traj[b:b + 1])
                    rewards.append(float(r.item()) if hasattr(r, "item") else float(r))
                except Exception:
                    rewards.append(0.0)
            return torch.tensor(rewards, device=device, dtype=torch.float32)

        # Integer step indices, matching resample(sampling_idx=step_idx). Using
        # linspace(1.0, 0.0, ...) here (as upstream pi05 does) truncates to
        # [1,0,0,...] and silently disables resampling.
        fkd = FKD(
            potential_type=fkd_config.get("potential_type", "max"),
            lmbda=fkd_config.get("lmbda", 10.0),
            num_particles=batch_size,
            adaptive_resampling=fkd_config.get("adaptive_resampling", True),
            resample_frequency=fkd_config.get("resample_frequency", 5),
            resampling_t_start=int(start_time * num_steps),
            resampling_t_end=num_steps,
            timesteps=torch.arange(num_steps + 1),
            reward_fn=reward_fn,
            reward_min_value=float("-inf"),
            device=device,
        )
        if len(fkd.t_to_index) != num_steps + 1:
            raise RuntimeError(
                f"FKD timestep collapse: {len(fkd.t_to_index)} unique indices for "
                f"{num_steps + 1} timesteps — resampling would silently no-op."
            )
        return fkd


def _reindex_particles(indices, past_key_values, prefix_pad_masks, state):
    """Reorder per-particle state to follow an FKD resample.

    Without this, particle i would keep denoising under particle j's language /
    image conditioning. It is a no-op while every batch row shares one prompt
    (which the single-prompt assert guarantees), but is applied defensively.
    """
    idx = indices.to(prefix_pad_masks.device)

    if isinstance(past_key_values, dict):
        for layer in past_key_values.values():
            if isinstance(layer, dict):
                for k in ("key_states", "value_states"):
                    if k in layer and torch.is_tensor(layer[k]):
                        layer[k] = layer[k][idx]

    return past_key_values, prefix_pad_masks[idx], state[idx]


def compute_guided_actions(cfg, task_description, processed_obs, image_key,
                           pi0_policy, sampler, keypoints, guidance_fn,
                           action_noise_std, batch_size,
                           global_step=0, current_stage=1):
    """Drop-in sibling of ``rl2_utils.compute_composed_actions``.

    Returns ``(actions, action_queue, w)`` with the same shapes, so the rollout
    call site is a plain if/else and everything downstream is untouched. ``w`` is
    always ``None``: there is no QAM composition under VLS.
    """
    action_queue = deque([], maxlen=cfg.n_action_steps)
    task_list = [task_description] * batch_size

    actions = sampler.sample(
        processed_obs=processed_obs,
        image_key=image_key,
        task_list=task_list,
        batch_size=batch_size,
        keypoints=keypoints,
        guidance_fn=guidance_fn,
        action_noise_std=action_noise_std,
        guide_scale=getattr(cfg, "vls_guide_scale", 80.0),
        diversity_scale=getattr(cfg, "vls_diversity_scale", 20.0),
        start_ratio=getattr(cfg, "vls_start_ratio", None),
        sigmoid_k=getattr(cfg, "vls_sigmoid_k", 25.0),
        sigmoid_x0=getattr(cfg, "vls_sigmoid_x0", 0.75),
        use_diversity=getattr(cfg, "vls_use_diversity", True),
        use_fkd=getattr(cfg, "vls_use_fkd", False),
        # FKD sub-config, defaults from VLS configs/config.yaml `main.fkd`.
        # NOTE: must not be None when use_fkd is set -- _init_fkd() returns
        # early on a None config, which would silently disable FK resampling.
        fkd_config={
            "potential_type": getattr(cfg, "vls_fkd_potential_type", "max"),
            "lmbda": getattr(cfg, "vls_fkd_lmbda", 10.0),
            "adaptive_resampling": getattr(cfg, "vls_fkd_adaptive_resampling", True),
            "resample_frequency": getattr(cfg, "vls_fkd_resample_frequency", 5),
        },
        global_step=global_step,
        current_stage=current_stage,
    )

    action_queue.extend(actions.transpose(0, 1))
    return actions, action_queue, None
