"""
Save one initial-frame PNG per task instruction, for use by
generate_simpler_rephrases_vlm.py (INITIAL_FRAME_DIR / find_matching_image).

Images are named after the exact instruction string returned by
env.get_language_instruction() (spaces -> underscores), matching the
lookup convention used in generate_simpler_rephrases_vlm.py.

Usage (from RL2_CoVer_VLA/simpler, with the SimplerEnv conda env active):
    export SAPIEN_EGL_ENV=1
    export MUJOCO_GL="egl"
    export LD_LIBRARY_PATH=/home/coder/cover-vla/nvidia_libs:$LD_LIBRARY_PATH
    export VK_ICD_FILENAMES=~/.vulkan/icd.d/nvidia_icd.json
    xvfb-run --auto-servernum -s "-screen 0 640x480x24" \
        python save_initial_frames.py --task_suites simpler_google_open_drawer simpler_google_close_drawer simpler_google_coke_vertical simpler_google_apple_in_drawer
"""
import argparse
from pathlib import Path

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from PIL import Image

from simpler_benchmark import task_map

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR.parents[1] / "bridge_verifier" / "initial_frame"

# All single-task google_robot suites, including top/bottom drawer variants.
DEFAULT_GOOGLE_ROBOT_TASK_SUITES = [
    "simpler_google_open_top_drawer",
    "simpler_google_open_middle_drawer",
    "simpler_google_open_bottom_drawer",
    "simpler_google_close_top_drawer",
    "simpler_google_close_middle_drawer",
    "simpler_google_close_bottom_drawer",
    # "simpler_google_apple_in_drawer",
    "simpler_google_coke_horizontal",
    "simpler_google_coke_vertical",
    "simpler_google_coke_standing",
]


def save_initial_frame(task_id: str, output_dir: Path) -> str:
    """Reset `task_id`'s env once and save its first frame. Returns the instruction text."""
    env = simpler_env.make(task_id, renderer_kwargs={"offscreen_only": True})
    is_google_robot = task_id.startswith("google_robot")

    if is_google_robot:
        # drawer_id / model_id (and thus the instruction) are only populated on reset
        obs, _ = env.reset(seed=0, options={"obj_init_options": {"episode_id": 0}})
    else:
        obs, _ = env.reset(seed=0)

    instruction = env.get_language_instruction()
    image = get_image_from_maniskill2_obs_dict(env, obs)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = instruction.replace(" ", "_") + ".png"
    save_path = output_dir / filename
    Image.fromarray(image).save(save_path)
    print(f"[{task_id}] instruction='{instruction}' -> {save_path}")

    env.close()
    return instruction


def main():
    parser = argparse.ArgumentParser(description="Save initial-frame PNGs for SimplerEnv task suites")
    parser.add_argument(
        "--task_suites",
        type=str,
        nargs="+",
        default=DEFAULT_GOOGLE_ROBOT_TASK_SUITES,
        help="Task suite names as registered in simpler_benchmark.task_map",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save PNGs into (default: bridge_verifier/initial_frame, "
        "matching generate_simpler_rephrases_vlm.py's INITIAL_FRAME_DIR)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    seen_instructions = {}
    for task_suite in args.task_suites:
        if task_suite not in task_map:
            print(f"WARNING: unknown task suite '{task_suite}', skipping")
            continue
        for task_id in task_map[task_suite]:
            instruction = save_initial_frame(task_id, output_dir)
            if instruction in seen_instructions:
                print(
                    f"    NOTE: instruction '{instruction}' was already saved from "
                    f"'{seen_instructions[instruction]}' (identical instruction text -> same filename)"
                )
            seen_instructions[instruction] = task_id


if __name__ == "__main__":
    main()
