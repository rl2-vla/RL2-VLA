#!/bin/bash
# NOTE: Activate the environment first (from repo root):
#   conda activate rl2-vls
#
# VLS (Vision-Language Steering) baseline: steers the FROZEN pi0 policy at
# inference time with VLM-synthesised differentiable rewards over 3D keypoints.
#
# VLS and RL2's CoVer machinery are ALTERNATIVE steering mechanisms, so this
# script disables the latter. run_simpler_eval_with_openpi.py asserts this at
# startup (before models load) rather than silently coercing the config:
#   use_verifier            False   VLS executes batch index 0, no re-selection
#   composed_samples*       0       no QAM compositional steering
#   use_failure_prediction  False   VLS is non-adaptive
#   lang_rephrase_num*      1       ONE fixed prompt per episode
#   lang_transform_type     rephrase  + rephrase_num=1 selects the JSON's
#                                     "original" field, i.e. the OOD base prompt
#                                     VLS is meant to recover from
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

# VLS needs both: OpenAI synthesises the guidance functions, Gemini does live
# stage recognition. They live in ~/.bashrc behind an interactive-shell guard,
# so source them explicitly if this runs non-interactively.
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (guidance synthesis)}"
: "${GOOGLE_API_KEY:?set GOOGLE_API_KEY (stage recognition; or pass --vls_use_vlm_stage_recognition False)}"

# ==========================================================================
# Eval config (TODO: Change here)
# ==========================================================================
GPU=0
SEEDS=(42 0 7)
NUM_TRIALS_PER_TASK=50

# Particle count for the steered sampler == VLS's sample_batch_size.
#   1  = VLS's shipped configs/config.yaml default (gradient guidance only;
#        RBF diversity and FK resampling self-disable below 2 particles)
#   5  = VLS's own run_main.sh
#   10 = the paper's ablation configs
ACTION_SAMPLES_PREFAIL=5

# Must stay 1 / 0: see the isolation asserts above.
LANG_REPHRASE_NUM_PREFAIL=1
COMPOSED_SAMPLES_PREFAIL=0

# ==========================================================================
# VLS hyperparameters (defaults copied verbatim from VLS configs/config.yaml)
# ==========================================================================
VLS_GUIDE_SCALE=10.0            # keypoint-gradient strength
VLS_DIVERSITY_SCALE=20.0        # RBF repulsion between particles (needs >1)
VLS_SIGMOID_K=25.0              # guidance decays as the stage nears its goal
VLS_SIGMOID_X0=0.75
VLS_USE_DIVERSITY=True
VLS_USE_FKD=True                # Feynman-Kac particle resampling (needs >1)
VLS_VLM_QUERY_LIMIT=50          # cap on live stage-recognition calls per episode
VLS_USE_VLM_STAGE_RECOGNITION=True

# Guidance cache. Guidance functions reference KEYPOINT INDICES, which are only
# meaningful for the task they were generated on, so the cache is per task:
#
#   <VLS_GUIDANCE_DIR>/<task_suite_name>/{metadata.json, stage*_guidance.txt, ...}
#
# Upstream VLS has no task lookup (main.py:343-347 takes a single
# cached_functions_dir) because run_main.sh runs ONE suite per invocation. This
# script loops over several, so the per-task slot is resolved explicitly.
#
# Set it and the first run synthesises + saves per task; later runs replay
# offline, deterministically and for free. Leave empty to regenerate every
# episode (upstream's default), which costs one OpenAI call per episode --
# at 50 trials x 4 tasks x 3 seeds that is 600 calls.
VLS_GUIDANCE_DIR=""

# Visualisation. Writes episode_N_{success,fail}_<camera>.mp4 with the sampled
# trajectories (one polyline per particle), numbered keypoints, and the
# Step/Stage/Guide/Norm_R/Sig_Str/Scale HUD -- same artifacts as VLS's own
# run_main.sh. Off by default: it costs time per step.
VLS_SAVE_VIDEO=True
VLS_VIZ_TRAJECTORY=False
VLS_VIZ_KEYPOINTS=True

# Log Directory
LOCAL_LOG_DIR="./experiments"

# Set to "IID" or "OOD" to select which task-suite type to evaluate.
TASK_SUITE_TYPE="IID"

# ==========================================================================
# Other config
# ==========================================================================

# Set the base directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
# Add repo root so bridge_verifier can be imported (go up 3 levels to cover-vla root)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# Add CoVer_VLA root so robot_utils can be imported
INFERENCE_ROOT="$REPO_ROOT/CoVer_VLA"
export PYTHONPATH="$REPO_ROOT:$INFERENCE_ROOT:$PYTHONPATH"
export PRISMATIC_DATA_ROOT=.

# HF pretrained checkpoint for INTACT Pi0 finetuned on Bridge-V2
PRETRAINED_CHECKPOINT="juexzz/INTACT-pi0-finetune-bridge"

if [[ "$TASK_SUITE_TYPE" == "IID" ]]; then
    TASK_SUITES=(
        simpler_put_eggplant_in_basket
        simpler_spoon_on_towel
        simpler_stack_cube
        simpler_carrot_on_plate
    )
else
    TASK_SUITES=(
        simpler_orange_juice_on_plate
        simpler_spoon_on_towel_google
        simpler_tape_measure_in_basket
        simpler_toy_dinosaur_on_towel
    )
fi

# ==========================================================================
# VLS (Vision-Language Steering)
# ==========================================================================
for seed in "${SEEDS[@]}"; do
    for task_suite in "${TASK_SUITES[@]}"; do
        CUDA_VISIBLE_DEVICES=$GPU python ../run_simpler_eval_with_openpi.py \
            --task_suite_name "$task_suite" \
            --lang_transform_type rephrase \
            --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
            --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
            --use_failure_prediction False \
            --lang_rephrase_num_prefail "$LANG_REPHRASE_NUM_PREFAIL" \
            --lang_rephrase_num 1 \
            --action_samples_prefail "$ACTION_SAMPLES_PREFAIL" \
            --composed_samples_prefail "$COMPOSED_SAMPLES_PREFAIL" \
            --composed_samples 0 \
            --use_verifier False \
            --use_vls True \
            --vls_guide_scale "$VLS_GUIDE_SCALE" \
            --vls_diversity_scale "$VLS_DIVERSITY_SCALE" \
            --vls_sigmoid_k "$VLS_SIGMOID_K" \
            --vls_sigmoid_x0 "$VLS_SIGMOID_X0" \
            --vls_use_diversity "$VLS_USE_DIVERSITY" \
            --vls_use_fkd "$VLS_USE_FKD" \
            --vls_vlm_query_limit "$VLS_VLM_QUERY_LIMIT" \
            --vls_use_vlm_stage_recognition "$VLS_USE_VLM_STAGE_RECOGNITION" \
            --vls_guidance_dir "$VLS_GUIDANCE_DIR" \
            --vls_save_video "$VLS_SAVE_VIDEO" \
            --vls_viz_trajectory "$VLS_VIZ_TRAJECTORY" \
            --vls_viz_keypoints "$VLS_VIZ_KEYPOINTS" \
            --seed "$seed" \
            --local_log_dir "$LOCAL_LOG_DIR" \
            --wandb_project VLS
    done
done
