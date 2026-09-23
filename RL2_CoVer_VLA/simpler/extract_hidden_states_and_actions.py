"""
Hidden State Extraction from Bridge TFRecords using PI0 (used to train QAM)

Loads Bridge dataset TFRecord files, forward passes each step through the PI0 policy,
and saves the extracted action_embeds (diffusion denoising hidden states) back into
new TFRecord files.

Checks:
- Hidden_states: shape: 1, 10, 5, 1024
- Hidden_states: dtype: float32
- VLA model: juexzz/INTACT-pi0-finetune-bridge

For paper: "RL2-VLA: Adaptive RL Latent Compositional Steering with Test-Time Scaling for Vision-Language-Action Models"
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# SIMPLER environment imports
from eval_utils import (
    convert_maniskill_with_bridge_adapter,
    create_simpler_adapter_wrapper,
    load_pi0_policy_compat,
    set_seed_everywhere,
)

# PI0 policy imports
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy


# =========================================================================================
# Helper: RPY -> Quaternion (wxyz)
# =========================================================================================

def rpy_to_quat(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z).
    """
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
    quat_xyzw = rot.as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return quat_wxyz


# =========================================================================================
# Helper: fractal relative -> absolute gripper action (visualization only)
# numpy port of INT-ACT/src/data/utils/data_utils.py::rel2abs_gripper_actions (the OXE
# rt1_dataset_transform used to build fractal's GT action at training time). Cast to float32
# to match TF's default dtype -- at the exact +-0.1 threshold, float64 promotion can flip the
# open/close classification for a value that float32 rounds to precisely 0.1.
# =========================================================================================

def rel2abs_gripper_actions(actions):
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute gripper
    actions (0 for closed, 1 for open). Assumes the first relative gripper is not redundant.
    """
    actions = np.asarray(actions, dtype=np.float32)
    opening_mask = actions < -0.1
    closing_mask = actions > 0.1
    thresholded = np.where(opening_mask, 1, np.where(closing_mask, -1, 0)).astype(np.int64)

    nonzero_idx = np.argmax(thresholded != 0) if np.any(thresholded != 0) else 0
    start = -1 * thresholded[nonzero_idx]
    if start == 0:
        start = 1  # no relative grasp in the whole trajectory -> assume open throughout

    new_actions = np.empty_like(thresholded)
    carry = start
    for i in range(len(thresholded)):
        if thresholded[i] != 0:
            carry = thresholded[i]
        new_actions[i] = carry

    return new_actions.astype(np.float32) / 2 + 0.5  # {-1,+1} -> {0,1}


# =========================================================================================
# Helper: Visualize policy actions vs GT actions across 7 dims x horizon steps
# =========================================================================================

def visualize_policy_vs_gt_actions(policy_actions, gt_actions, step_idx, episode_idx,
                                    task_description, save_path=None):
    """
    Bar chart comparing policy actions vs GT actions for each of 7 dims over horizon steps.

    Args:
        policy_actions: (H, 7) float32 — unnormalized policy predictions
        gt_actions:     (H, 7) float32 — GT next actions from TFRecord
                        [dx, dy, dz, droll, dpitch, dyaw, gripper]
        step_idx:       current step index within the episode
        episode_idx:    current episode index
        task_description: instruction string for plot title
        save_path:      if provided, save figure here; otherwise call plt.show()
    """
    dim_labels = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']
    horizon = policy_actions.shape[0]
    x = np.arange(horizon)
    width = 0.35

    fig, axes = plt.subplots(1, 7, figsize=(20, 3))
    fig.suptitle(
        f"Policy vs GT Actions | Ep {episode_idx}  Step {step_idx}\n{task_description[:100]}",
        fontsize=8,
    )

    for dim_idx, ax in enumerate(axes):
        ax.bar(x - width / 2, policy_actions[:, dim_idx], width, label='policy', alpha=0.85, color='steelblue')
        ax.bar(x + width / 2, gt_actions[:, dim_idx], width, label='GT', alpha=0.85, color='tomato')
        ax.set_title(dim_labels[dim_idx], fontsize=8)
        ax.set_xlabel('horizon step', fontsize=7)
        ax.set_xticks(x)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color='black', linewidth=0.5)
        if dim_idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved action comparison plot: {save_path}")
    else:
        plt.show()


# =========================================================================================
# Helper: Extract action_embeds from a single TFRecord step observation
# =========================================================================================

def extract_embeds_from_obs(pi0_policy, preprocess_adapter, raw_img, eef_pos_8d,
                            task_description, cfg, action_noise_std):
    """
    Extract action_embeds from a single TFRecord step observation.

    Args:
        pi0_policy: Loaded PI0Policy model
        preprocess_adapter: BridgeSimplerAdapter from create_bridge_adapter_wrapper
        raw_img: (H, W, 3) uint8 numpy image decoded from TFRecord
        eef_pos_8d: (8,) array [x, y, z, qw, qx, qy, qz, gripper] — SimplerEnv eef_pos format
        task_description: Instruction string from TFRecord
        cfg: namespace/object with policy_batch_inference_size, lang_rephrase_num, n_action_steps
        action_noise_std: Noise std for action sampling

    Returns:
        action_embeds: np.ndarray (B, num_diffusion_steps, H, hidden_dim)
    """
    # Reconstruct the obs dict that preprocess_proprio expects:
    # preprocess_proprio reads obs["agent"]["eef_pos"] — same structure as SimplerEnv
    obs = {
        'agent': {
            'eef_pos': eef_pos_8d  # (8,): [x, y, z, qw, qx, qy, qz, gripper]
        }
    }
    obs_for_adapter = {
        'observation.images.top': raw_img,
        'observation.state': obs,
        'task': task_description
    }
    processed_obs = preprocess_adapter.preprocess(obs_for_adapter)

    # Move to policy device  
    policy_device = torch.device(pi0_policy.config.device)
    processed_obs = {
        k: (v.to(device=policy_device) if isinstance(v, torch.Tensor) else v)
        for k, v in processed_obs.items()
    }

    # Get image feature key from policy config  
    image_feature_keys = list(pi0_policy.config.image_features.keys())
    image_key = image_feature_keys[0]

    # Create batch of language instructions  
    batch_size = cfg.policy_batch_inference_size * cfg.lang_rephrase_num

    # Repeat same instruction for batch  
    task_list = [task_description] * batch_size

    # Create batch observation dict  
    batch_image = processed_obs['observation.images.top'].repeat(batch_size, 1, 1, 1)
    batch_state = processed_obs['observation.state'].repeat(batch_size, 1)

    observation = {
        image_key: batch_image,
        "observation.state": batch_state,
        "task": task_list,
    }

    # Call select_action  
    pi0_policy.reset()
    with torch.no_grad():
        output_action_queue, action_embeds = pi0_policy.select_action(
            observation, noise_std=action_noise_std, return_action_embeds=True
        )

    return action_embeds, output_action_queue  # (B, num_diffusion_steps, H, hidden_dim), list of (B, 7)


# =========================================================================================
# Main Processing Function
# =========================================================================================

def process_tfrecords(
    pretrained_checkpoint,
    input_dir,
    output_dir,
    split,
    image_key,
    selected_files,
    seed,
    policy_batch_inference_size,
    lang_rephrase_num,
    n_action_steps,
    action_ensemble_temp,
    model_family,
    embodiment="widowx",
    visualize=False,
):
    """
    Load PI0 model and extract action_embeds from each step in TFRecord files.
    """
    print("=" * 80)
    print("======================== Loading model ========================")
    print("=" * 80)

    # Seeding
    set_seed_everywhere(seed)

    is_google_robot = embodiment == "google_robot"

    # Build a minimal cfg-like namespace so extract_embeds_from_obs can use cfg.* attributes
    class Cfg:
        pass
    cfg = Cfg()
    cfg.policy_batch_inference_size = policy_batch_inference_size
    cfg.lang_rephrase_num = lang_rephrase_num
    cfg.n_action_steps = n_action_steps
    cfg.action_ensemble_temp = action_ensemble_temp
    cfg.model_family = model_family
    cfg.embodiment = embodiment

    # Set action un-normalization key
    if cfg.model_family == "prismatic":
        cfg.unnorm_key = "bridge_dataset"
    elif is_google_robot:
        cfg.unnorm_key = "fractal"
    else:
        cfg.unnorm_key = "bridge_orig"

    # Initialize PI0 policy
    print(f"Loading model from {pretrained_checkpoint}...")
    if is_google_robot:
        # fractal checkpoints ship a newer-lerobot config.json
        pi0_policy = load_pi0_policy_compat(pretrained_checkpoint)
    else:
        pi0_policy = PI0Policy.from_pretrained(pretrained_checkpoint)

    if torch.cuda.is_available():
        pi0_policy.to("cuda")
        pi0_policy.config.device = "cuda"

    pi0_policy.config.n_action_steps = int(cfg.n_action_steps)
    print(f"PI0Policy device: {pi0_policy.config.device}")

    ensemble_model = None

    # Create adapter for preprocessing (embodiment-specific:
    # widowx -> BridgeSimplerAdapter, google_robot -> EDRSimplerAdapterRaw)
    action_queue = None
    if not hasattr(pi0_policy, '_preprocess_adapter'):
        pi0_policy._preprocess_adapter = create_simpler_adapter_wrapper(embodiment, cfg.action_ensemble_temp)
    preprocess_adapter = pi0_policy._preprocess_adapter
    preprocess_adapter.collect_latents = True

    # Action noise for batch inference
    action_noise_std = 1.0

    print("=" * 80)
    print("======================== Extracting action_embeds ========================")
    print("=" * 80)
    print(f"Input:          {input_dir}")
    print(f"Output:         {output_dir}")
    print(f"Split:          {split}")
    print(f"Image key:      {image_key}")
    print(f"Batch size:     {policy_batch_inference_size} x {lang_rephrase_num} = {policy_batch_inference_size * lang_rephrase_num}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(selected_files)} files")
    for f in selected_files[:5]:
        print(f"  {os.path.basename(f)}")
    if len(selected_files) > 5:
        print(f"  ... and {len(selected_files) - 5} more")

    total_records = 0
    total_steps = 0
    first_step_shown = False

    for input_file in selected_files:
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        print(f"\nProcessing: {os.path.basename(input_file)}")

        dataset = tf.data.TFRecordDataset(input_file)
        writer = tf.io.TFRecordWriter(output_file)

        file_record_count = 0
        file_steps_count = 0

        for episode_idx, raw_record in enumerate(dataset):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            features = example.features.feature

            instruction_key = (
                'steps/observation/natural_language_instruction' if is_google_robot
                else 'steps/language_instruction'
            )

            num_steps = len(features[f'steps/observation/{image_key}'].bytes_list.value)
            first_language_instruction_bytes = features[instruction_key].bytes_list.value[0]
            episode_task_description = first_language_instruction_bytes.decode('utf-8')

            print(f"\n  Episode {episode_idx}: {num_steps} steps | Task: '{episode_task_description}'")

            # Extract per-episode proprio arrays (layout differs by embodiment)
            if is_google_robot:
                # (N, 7): [x, y, z, qx, qy, qz, qw] (xyzw quat, per OXE rt1_dataset_transform)
                poses = np.array(features['steps/observation/base_pose_tool_reached'].float_list.value)
                poses = poses.reshape(num_steps, -1)
                # (N,): 0 = open, 1 = closed
                gripper_closed = np.array(features['steps/observation/gripper_closed'].float_list.value)
            else:
                # Extract states for the whole episode
                states = np.array(features['steps/observation/state'].float_list.value)
                states = states.reshape(num_steps, -1)

            # Extract GT actions for the whole episode, in the same [dx,dy,dz,droll,dpitch,dyaw,gripper]
            # layout for both embodiments (gripper in [0,1]) -- visualization only, never saved to the
            # output TFRecord. Like Bridge's raw steps/action (compared as-is against policy_actions'
            # axis-angle rotation dims), fractal's rotation_delta is used as-is too -- same approximation
            # Bridge already accepts, just assembled from three separate fields instead of one flat array.
            if visualize:
                if is_google_robot:
                    world_vector = np.array(features['steps/action/world_vector'].float_list.value).reshape(num_steps, -1)
                    rotation_delta = np.array(features['steps/action/rotation_delta'].float_list.value).reshape(num_steps, -1)
                    gripper_closedness_action = np.array(features['steps/action/gripper_closedness_action'].float_list.value)
                    # relative (+1 closing/-1 opening/0 no-change) -> absolute (0=closed,1=open),
                    # matching policy_actions' gripper convention -- unlike Bridge's gripper slot
                    # (already absolute), fractal's raw field is a relative delta.
                    gt_gripper_open = rel2abs_gripper_actions(gripper_closedness_action)
                    gt_actions_episode = np.concatenate(
                        [world_vector, rotation_delta, gt_gripper_open[:, None]], axis=1
                    ).astype(np.float32)
                else:
                    gt_actions_episode = np.array(features['steps/action'].float_list.value).reshape(num_steps, -1).astype(np.float32)

            for step_idx in tqdm(range(num_steps), desc=f"    Steps", leave=False):
                # Decode image from TFRecord
                image_bytes = features[f'steps/observation/{image_key}'].bytes_list.value[step_idx]
                raw_img = tf.io.decode_image(image_bytes, channels=3).numpy()  # (H, W, 3) uint8

                # Decode instruction
                instruction_bytes = features[instruction_key].bytes_list.value[step_idx]
                task_description = instruction_bytes.decode('utf-8')

                if is_google_robot:
                    # base_pose_tool_reached is [x,y,z, qx,qy,qz,qw] (xyzw); SimplerEnv eef_pos wants
                    # [x,y,z, qw,qx,qy,qz, gripper_width]. gripper_closed is 0=open/1=closed (dataset
                    # convention); SimplerEnv width is 0=close/1=open, hence the inversion.
                    pose = poses[step_idx]  # (7,): [x, y, z, qx, qy, qz, qw]
                    position = pose[:3]
                    qx, qy, qz, qw = pose[3:7]
                    quat_wxyz = np.array([qw, qx, qy, qz])
                    gripper_width = 1.0 - gripper_closed[step_idx:step_idx + 1]
                    eef_pos_8d = np.concatenate([position, quat_wxyz, gripper_width])  # (8,)
                    tfrecord_state = pose
                else:
                    # Convert TFRecord state (x, y, z, roll, pitch, yaw, gripper) ->
                    # SimplerEnv eef_pos format (x, y, z, qw, qx, qy, qz, gripper)
                    tfrecord_state = states[step_idx]  # (7,): [x, y, z, roll, pitch, yaw, gripper]
                    position = tfrecord_state[:3]
                    roll, pitch, yaw = tfrecord_state[3:6]
                    gripper = tfrecord_state[6:7]
                    quat_wxyz = rpy_to_quat(roll, pitch, yaw)
                    eef_pos_8d = np.concatenate([position, quat_wxyz, gripper])  # (8,)

                # Extract action_embeds 
                action_embeds, output_action_queue = extract_embeds_from_obs(
                    pi0_policy=pi0_policy,
                    preprocess_adapter=preprocess_adapter,
                    raw_img=raw_img,
                    eef_pos_8d=eef_pos_8d,
                    task_description=task_description,
                    cfg=cfg,
                    action_noise_std=action_noise_std,
                )  # (B, num_diffusion_steps, H, hidden_dim), list of (B, 7)
                print(action_embeds.shape)
                # Convert to float32 for storage 
                action_embeds_f32 = action_embeds.astype(np.float32)

                # Serialize and append to TFRecord features 
                serialized_embeds = tf.io.serialize_tensor(tf.constant(action_embeds_f32))
                features['steps/observation/hidden_states'].bytes_list.value.append(
                    serialized_embeds.numpy()
                )

                # Unnormalize and convert actions via the embodiment-specific adapter (verifier_action=False)
                # output_action_queue[i] shape: (B, 7); take first sample (B=1)
                processed_actions = []
                for i in range(cfg.n_action_steps):
                    action_1x7 = output_action_queue[i].cpu().numpy()[0:1]  # (1, 7)
                    processed_action = convert_maniskill_with_bridge_adapter(
                        action_1x7, verifier_action=False, action_ensemble_temp=cfg.action_ensemble_temp,
                        embodiment=cfg.embodiment,
                    )  # (7,)
                    processed_actions.append(processed_action)
                policy_actions = np.stack(processed_actions).astype(np.float32)  # (4, 7)
                assert policy_actions.shape == (cfg.n_action_steps, 7)

                if visualize:
                    # Gripper-remapped copy for the GT comparison plot only -- never written to the
                    # TFRecord, so the saved policy_actions stays untouched. SimplerEnv gripper sign
                    # convention is opposite between adapters: BridgeSimplerAdapter outputs +1=open/
                    # -1=close, EDRSimplerAdapterRaw outputs -1=open/+1=close (see postprocess_gripper
                    # in eval_utils.py) -- each needs its own formula to land on [0,1]=[close,open].
                    policy_actions_for_viz = policy_actions.copy()
                    if is_google_robot:
                        policy_actions_for_viz[:, -1] = (1 - policy_actions_for_viz[:, -1]) / 2
                    else:
                        policy_actions_for_viz[:, -1] = (policy_actions_for_viz[:, -1] + 1) / 2

                # Save policy_actions per step
                serialized_actions = tf.io.serialize_tensor(tf.constant(policy_actions))
                features['steps/observation/policy_actions'].bytes_list.value.append(
                    serialized_actions.numpy()
                )

                # Show first step as example
                if not first_step_shown:
                    print("\n" + "=" * 80)
                    print("Example: First step with action_embeds")
                    print("=" * 80)
                    print(f"Task:              '{task_description}'")
                    print(f"Image shape:       {raw_img.shape}")
                    print(f"State shape:       {tfrecord_state.shape}")
                    print(f"action_embeds shape: {action_embeds_f32.shape}")
                    print(f"  -> (B={action_embeds_f32.shape[0]}, num_diffusion_steps={action_embeds_f32.shape[1]}, H={action_embeds_f32.shape[2]}, hidden_dim={action_embeds_f32.shape[3]})")
                    print(f"action_embeds dtype: {action_embeds_f32.dtype}")
                    print(f"policy_actions shape: {policy_actions.shape}  -> (n_action_steps={policy_actions.shape[0]}, action_dim={policy_actions.shape[1]})")
                    print(f"policy_actions dtype: {policy_actions.dtype}")
                    print("=" * 80 + "\n")
                    first_step_shown = True

                if visualize:
                    gt_end = min(step_idx + cfg.n_action_steps, num_steps)
                    gt_slice = gt_actions_episode[step_idx:gt_end]
                    if gt_slice.shape[0] < cfg.n_action_steps:
                        pad = np.repeat(gt_slice[-1:], cfg.n_action_steps - gt_slice.shape[0], axis=0)
                        gt_slice = np.concatenate([gt_slice, pad], axis=0)
                    gt_actions_for_viz = gt_slice  # (4, 7) — gripper in [0,1], matches policy_actions_for_viz
                    viz_dir = os.path.join(output_dir, "action_comparisons")
                    os.makedirs(viz_dir, exist_ok=True)
                    viz_path = os.path.join(viz_dir, f"ep{episode_idx:04d}_step{step_idx:04d}.png")
                    visualize_policy_vs_gt_actions(
                        policy_actions_for_viz, gt_actions_for_viz,
                        step_idx=step_idx, episode_idx=episode_idx,
                        task_description=task_description,
                        save_path=viz_path,
                    )

                total_steps += 1
                file_steps_count += 1

            print(f"    Episode {file_record_count}: {num_steps} steps -> action_embeds added")

            writer.write(example.SerializeToString())
            file_record_count += 1
            total_records += 1

        writer.close()
        print(f"  Wrote {file_record_count} episodes ({file_steps_count} steps) to {os.path.basename(output_file)}")

    print(f"\n{'=' * 80}")
    print(f"Processing complete!")
    print(f"Total episodes processed: {total_records}")
    print(f"Total steps processed:    {total_steps}")
    print(f"Output directory:         {output_dir}")
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith('.tfrecord')
    )
    print(f"Total output size:        {total_size / 1e9:.2f} GB")
    print(f"{'=' * 80}")


# =========================================================================================
# Main
# =========================================================================================

if __name__ == "__main__":
    # Fixed settings
    BASE_DIR = '/home/coder/hdd/oxe_ds_new_v2'
    OUTPUT_BASE_DIR = '/home/coder/hdd/oxe_ds_with_action_embeds_pizero_fractal_haomingsong_seed_42_w_ACTIONS'

    # Model settings
    POLICY_BATCH_INFERENCE_SIZE = 1
    LANG_REPHRASE_NUM = 1
    N_ACTION_STEPS = 4
    ACTION_ENSEMBLE_TEMP = -0.8
    MODEL_FAMILY = "pi0"
    SEED = 42
    VISUALIZE = False

    # ============================================
    # INTERACTIVE: Select dataset
    # ============================================
    print("\n" + "=" * 80)
    print("DATASET SELECTION")
    print("=" * 80)

    # (dataset_name, dataset_path, dataset_prefix, image_key, pretrained_checkpoint, embodiment)
    datasets = {
        '1': ('bridge_dataset', 'bridge_dataset/1.0.0', 'bridge',
              'image_0', "juexzz/INTACT-pi0-finetune-bridge", "widowx"),
        '2': ('fractal20220817_data', 'fractal20220817_data/0.1.0', 'fractal20220817_data',
              'image', "HaomingSong/lerobot-pi0-fractal", "google_robot"),
    }

    print("\nAvailable datasets:")
    for key, (name, path, prefix, image_key, checkpoint, embodiment) in datasets.items():
        exists = "✓" if os.path.exists(os.path.join(BASE_DIR, path)) else "✗"
        print(f"  {key}. {name} [{exists}]")

    while True:
        dataset_choice = input("\nSelect dataset (1 or 2): ").strip()
        if dataset_choice in datasets:
            dataset_name, dataset_path, dataset_prefix, IMAGE_KEY, PRETRAINED_CHECKPOINT, EMBODIMENT = datasets[dataset_choice]
            break
        print("Error: Please enter 1 or 2")

    print(f"\nSelected dataset: {dataset_name}")

    # ============================================
    # INTERACTIVE: Select split
    # ============================================
    print("\n" + "=" * 80)
    print("SPLIT SELECTION")
    print("=" * 80)

    full_path = os.path.join(BASE_DIR, dataset_path)
    exists = "✓" if os.path.exists(full_path) else "✗"
    print(f"\nDataset: {dataset_name} [{exists}]")

    print("\nAvailable splits:")
    print("  1. train")
    print("  2. val")

    while True:
        split_choice = input("\nSelect split (1 or 2, default: 1): ").strip() or '1'
        if split_choice == '1':
            SPLIT = 'train'
            break
        elif split_choice == '2':
            SPLIT = 'val'
            break
        print("Error: Please enter 1 or 2")

    print(f"Selected split: {SPLIT}")

    INPUT_DIR = os.path.join(BASE_DIR, dataset_path)
    OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, dataset_path)

    # ============================================
    # INTERACTIVE: Select TFRecord range
    # ============================================
    print("\n" + "=" * 80)
    print("TFRECORD RANGE SELECTION")
    print("=" * 80)

    tfrecord_pattern = os.path.join(INPUT_DIR, f'{dataset_prefix}-{SPLIT}.tfrecord-*')
    all_files = sorted(glob.glob(tfrecord_pattern))

    if not all_files:
        print(f"ERROR: No TFRecord files found matching: {tfrecord_pattern}")
        exit(1)

    print(f"\nExample filename: {os.path.basename(all_files[0])}")

    file_indices = []
    for f in all_files:
        basename = os.path.basename(f)
        match = re.search(r'tfrecord-(\d+)-of-', basename)
        if match:
            file_indices.append(int(match.group(1)))
        else:
            print(f"Warning: Could not parse index from: {basename}")

    if not file_indices:
        print(f"ERROR: Could not parse file indices from filenames")
        exit(1)

    min_idx = min(file_indices)
    max_idx = max(file_indices)
    total_files = len(file_indices)

    print(f"\nFound {total_files} TFRecord files")
    print(f"Index range: {min_idx:05d} to {max_idx:05d}")
    print(f"First file: {os.path.basename(all_files[0])}")
    print(f"Last file:  {os.path.basename(all_files[-1])}")

    print("\n" + "-" * 80)
    print("Enter the range of TFRecord files to process (inclusive)")
    print(f"Valid range: {min_idx} to {max_idx}")
    print("-" * 80)

    while True:
        try:
            start_idx = int(input(f"Start index (default: {min_idx}): ") or str(min_idx))
            end_idx = int(input(f"End index (default: {max_idx}): ") or str(max_idx))
            if start_idx < min_idx or start_idx > max_idx:
                print(f"Error: Start index must be between {min_idx} and {max_idx}")
                continue
            if end_idx < min_idx or end_idx > max_idx:
                print(f"Error: End index must be between {min_idx} and {max_idx}")
                continue
            if start_idx > end_idx:
                print(f"Error: Start index must be <= end index")
                continue
            break
        except ValueError:
            print("Error: Please enter valid integers")
            continue

    index_to_file = {}
    for f in all_files:
        match = re.search(r'tfrecord-(\d+)-of-', os.path.basename(f))
        if match:
            index_to_file[int(match.group(1))] = f

    selected_files = [index_to_file[idx] for idx in range(start_idx, end_idx + 1) if idx in index_to_file]

    print("\n" + "=" * 80)
    print(f"SELECTED RANGE: {start_idx:05d} to {end_idx:05d}")
    print(f"Processing {len(selected_files)} files")
    print("=" * 80)

    confirm = input("\nProceed with processing? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Aborted by user.")
        exit(0)

    print("\n" + "=" * 80)
    print("Starting processing...")
    print("=" * 80)

    process_tfrecords(
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        split=SPLIT,
        image_key=IMAGE_KEY,
        selected_files=selected_files,
        seed=SEED,
        policy_batch_inference_size=POLICY_BATCH_INFERENCE_SIZE,
        lang_rephrase_num=LANG_REPHRASE_NUM,
        n_action_steps=N_ACTION_STEPS,
        action_ensemble_temp=ACTION_ENSEMBLE_TEMP,
        model_family=MODEL_FAMILY,
        embodiment=EMBODIMENT,
        visualize=VISUALIZE,
    )