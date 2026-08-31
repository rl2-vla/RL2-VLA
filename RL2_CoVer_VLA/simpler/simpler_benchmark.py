from simpler_utils import get_simpler_env

BENCHMARK_MAPPING = {}


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    # Create an instance to get the name attribute
    instance = target_class()
    BENCHMARK_MAPPING[instance.name.lower()] = target_class


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


###

task_map = {
    "simpler_widowx": [
        "widowx_put_eggplant_in_basket",
        "widowx_spoon_on_towel",
        "widowx_stack_cube",
        "widowx_carrot_on_plate",
    ],
    "simpler_stack_cube": [
        "widowx_stack_cube",
    ],
    "simpler_put_eggplant_in_basket": [
        "widowx_put_eggplant_in_basket",
    ],
    "simpler_spoon_on_towel": [
        "widowx_spoon_on_towel",
    ],
    "simpler_carrot_on_plate": [
        "widowx_carrot_on_plate",
    ],
    "simpler_redbull_on_plate": [
        "widowx_redbull_on_plate",
    ],
    "simpler_tennis_ball_in_basket": [
        "widowx_tennis_ball_in_basket",
    ],
    "simpler_zucchini_on_towel": [
        "widowx_zucchini_on_towel",
    ],
    "simpler_ood":[
        "widowx_redbull_on_plate",
        "widowx_zucchini_on_towel",
        "widowx_tennis_ball_in_basket",
        # "widowx_toy_dinosaur_on_towel",
    ],
    # NEW STUFF
    "simpler_carrot_on_plate_unseen_lighting": [
        "widowx_carrot_on_plate_unseen_lighting",
    ],
    "simpler_spoon_on_towel_new_table_cloth": [
        "widowx_spoon_on_towel_new_table_cloth",
    ],
    "simpler_spoon_on_towel_google": [
        "widowx_spoon_on_towel_google",
    ],
    "simpler_tape_measure_in_basket": [
        "widowx_tape_measure_in_basket",
    ],
    "simpler_toy_dinosaur_on_towel": [
        "widowx_toy_dinosaur_on_towel",
    ],
    "simpler_stapler_on_paper": [
        "widowx_stapler_on_paper",
    ],
    "simpler_carrot_on_ramekin": [
        "widowx_carrot_on_ramekin_clean",
    ],
    "simpler_coke_can_on_ramekin": [
        "widowx_coke_can_on_ramekin_clean",
    ],
    "simpler_cube_on_plate": [
        "widowx_cube_on_plate_clean",
    ],
    "simpler_coke_can_on_plate": [
        "widowx_coke_can_on_plate_clean",
    ],
    "simpler_pepsi_on_plate": [
        "widowx_pepsi_on_plate_clean",
    ],
    "simpler_orange_juice_on_plate": [
        "widowx_orange_juice_on_plate_clean",
    ],
    "simpler_nut_on_wheel": [
        "widowx_nut_on_wheel_clean",
    ],

    # =========================== Google Robot (fractal) task suites ===========================
    # Only tasks with a per-episode-deterministic instruction are registered.
    "simpler_google_robot": [
        "google_robot_pick_horizontal_coke_can",
        "google_robot_pick_vertical_coke_can",
        "google_robot_pick_standing_coke_can",
        "google_robot_open_top_drawer",
        "google_robot_open_middle_drawer",
        "google_robot_open_bottom_drawer",
        "google_robot_close_top_drawer",
        "google_robot_close_middle_drawer",
        "google_robot_close_bottom_drawer",
        "google_robot_place_apple_in_closed_top_drawer",
    ],
    "simpler_google_coke_horizontal": ["google_robot_pick_horizontal_coke_can"],
    "simpler_google_coke_vertical": ["google_robot_pick_vertical_coke_can"],
    "simpler_google_coke_standing": ["google_robot_pick_standing_coke_can"],
    "simpler_google_open_drawer": [
        "google_robot_open_top_drawer",
        "google_robot_open_middle_drawer",
        "google_robot_open_bottom_drawer",
    ],
    "simpler_google_close_drawer": [
        "google_robot_close_top_drawer",
        "google_robot_close_middle_drawer",
        "google_robot_close_bottom_drawer",
    ],
    "simpler_google_open_top_drawer": ["google_robot_open_top_drawer"],
    "simpler_google_open_middle_drawer": ["google_robot_open_middle_drawer"],
    "simpler_google_open_bottom_drawer": ["google_robot_open_bottom_drawer"],
    "simpler_google_close_top_drawer": ["google_robot_close_top_drawer"],
    "simpler_google_close_middle_drawer": ["google_robot_close_middle_drawer"],
    "simpler_google_close_bottom_drawer": ["google_robot_close_bottom_drawer"],
    "simpler_google_apple_in_drawer": [
        "google_robot_place_apple_in_closed_top_drawer",
    ],

}

class Benchmark:
    def _make_benchmark(self):
        self.tasks = task_map[self.name]

    def get_task(self, i):
        return self.tasks[i]

    def make(self, *args, **kwargs):
        return self.env_fn(*args, **kwargs)

    @property
    def n_tasks(self):
        return len(self.tasks)


class SimplerBenchmark(Benchmark):
    embodiment = "widowx"

    def __init__(self):
        super().__init__()
        self.env_fn = get_simpler_env
        self.state_dim = 7


class GoogleRobotSimplerBenchmark(SimplerBenchmark):
    """SimplerEnv task suites running on the Google Robot (fractal) embodiment."""
    embodiment = "google_robot"


@register_benchmark
class SIMPLER_WIDOWX(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_widowx"
        self._make_benchmark()


@register_benchmark
class SIMPLER_WIDOWX_CUBE(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_stack_cube"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_EGGPLANT(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_put_eggplant_in_basket"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_SPOON(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_spoon_on_towel"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_CARROT(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_carrot_on_plate"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_REDBULL(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_redbull_on_plate"
        self._make_benchmark()
        
        
@register_benchmark
class SIMPLER_WIDOWX_TENNIS_BALL(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_tennis_ball_in_basket"
        self._make_benchmark()
        
@register_benchmark
class SIMPLER_WIDOWX_CARROT_UNSEEN_LIGHTING(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_carrot_on_plate_unseen_lighting"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_TOY_DINOSAUR(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_toy_dinosaur_on_towel"
        self._make_benchmark()
        
@register_benchmark
class SIMPLER_WIDOWX_ZUCCHINI(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_zucchini_on_towel"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_OOD(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_ood"
        self._make_benchmark()

@register_benchmark
class SIMPLER_SPOON_ON_TOWEL_NEW_TABLE_CLOTH(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_spoon_on_towel_new_table_cloth"
        self._make_benchmark()

@register_benchmark
class SIMPLER_SPOON_ON_TOWEL_GOOGLE(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_spoon_on_towel_google"
        self._make_benchmark()

@register_benchmark
class SIMPLER_TAPE_MEASURE_IN_BASKET(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_tape_measure_in_basket"
        self._make_benchmark()

@register_benchmark
class SIMPLER_STAPLER_ON_PAPER(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_stapler_on_paper"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_CARROT_ON_RAMEKIN(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_carrot_on_ramekin"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_COKE_CAN_ON_RAMEKIN(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_coke_can_on_ramekin"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_CUBE_ON_PLATE(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_cube_on_plate"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_COKE_CAN_ON_PLATE(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_coke_can_on_plate"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_PEPSI_ON_PLATE(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_pepsi_on_plate"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_ORANGE_JUICE_ON_PLATE(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_orange_juice_on_plate"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_NUT_ON_WHEEL(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_nut_on_wheel"
        self._make_benchmark()


# =============================== Google Robot (fractal) ===============================

@register_benchmark
class SIMPLER_GOOGLE_ROBOT(GoogleRobotSimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_google_robot"
        self._make_benchmark()

@register_benchmark
class SIMPLER_GOOGLE_COKE_HORIZONTAL(GoogleRobotSimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_google_coke_horizontal"
        self._make_benchmark()

@register_benchmark
class SIMPLER_GOOGLE_COKE_VERTICAL(GoogleRobotSimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_google_coke_vertical"
        self._make_benchmark()

@register_benchmark
class SIMPLER_GOOGLE_COKE_STANDING(GoogleRobotSimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_google_coke_standing"
        self._make_benchmark()

@register_benchmark
class SIMPLER_GOOGLE_OPEN_DRAWER(GoogleRobotSimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_google_open_drawer"
        self._make_benchmark()

@register_benchmark
class SIMPLER_GOOGLE_CLOSE_DRAWER(GoogleRobotSimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_google_close_drawer"
        self._make_benchmark()

@register_benchmark
class SIMPLER_GOOGLE_OPEN_TOP_DRAWER(GoogleRobotSimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_google_open_top_drawer"
        self._make_benchmark()

@register_benchmark
class SIMPLER_GOOGLE_OPEN_MIDDLE_DRAWER(GoogleRobotSimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_google_open_middle_drawer"
        self._make_benchmark()

@register_benchmark
class SIMPLER_GOOGLE_OPEN_BOTTOM_DRAWER(GoogleRobotSimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_google_open_bottom_drawer"
        self._make_benchmark()

@register_benchmark
class SIMPLER_GOOGLE_CLOSE_TOP_DRAWER(GoogleRobotSimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_google_close_top_drawer"
        self._make_benchmark()

@register_benchmark
class SIMPLER_GOOGLE_CLOSE_MIDDLE_DRAWER(GoogleRobotSimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_google_close_middle_drawer"
        self._make_benchmark()

@register_benchmark
class SIMPLER_GOOGLE_CLOSE_BOTTOM_DRAWER(GoogleRobotSimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_google_close_bottom_drawer"
        self._make_benchmark()

@register_benchmark
class SIMPLER_GOOGLE_APPLE_IN_DRAWER(GoogleRobotSimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_google_apple_in_drawer"
        self._make_benchmark()