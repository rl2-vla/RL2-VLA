"""
Environment Adapters for VLS steering.

Ported from the upstream VLS repo (`core/env_adapters/`). Upstream supports
CALVIN (PyBullet) and LIBERO (MuJoCo/Robosuite); this port targets SIMPLER
(ManiSkill2_real2sim / SAPIEN) only.

Usage:
    from vls.core.env_adapters import create_adapter

    adapter = create_adapter("simpler", env_config, env=env)
"""

from .base_adapter import BaseEnvAdapter, Pose3D, CameraParams, TrackedObject, InteractableObject
from .simpler_adapter import SimplerAdapter

# VLS-PORT: CalvinAdapter / LiberoAdapter imports removed. They import pybullet
# and libero/robosuite at module scope, neither of which is installed in the
# RL2-VLA environment, so importing them here would break `import vls`.
# from .calvin_adapter import CalvinAdapter
# from .libero_adapter import LiberoAdapter
# (create_calvin_env() dropped for the same reason.)


def create_adapter(backend: str, env_config: dict, **kwargs) -> BaseEnvAdapter:
    """
    Factory function to create the appropriate adapter.

    Args:
        backend: Only "simpler" is supported in this port.
        env_config: Environment configuration dict
        **kwargs: Passed through to the adapter (notably `env=<gym env>`)

    Returns:
        BaseEnvAdapter instance
    """
    backend = backend.lower()

    if backend == "simpler":
        # The rollout script owns the env (it has reset options the adapter
        # should not know about), so it is passed in rather than constructed.
        env = kwargs.pop("env", None)
        if env is None:
            raise ValueError("create_adapter('simpler', ...) requires env=<gym env>")
        return SimplerAdapter(env, env_config, **kwargs)

    raise ValueError(f"Unknown backend: {backend}. Supported: simpler")


__all__ = [
    "BaseEnvAdapter",
    "Pose3D",
    "CameraParams",
    "TrackedObject",
    "InteractableObject",
    "SimplerAdapter",
    "create_adapter",
]
