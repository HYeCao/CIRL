from importlib import import_module


def _register(mapping, exp_name, module_name):
    try:
        module = import_module(f"experiments.{module_name}.config")
    except ModuleNotFoundError as exc:
        expected = f"experiments.{module_name}"
        if exc.name == expected or exc.name.startswith(f"{expected}."):
            return
        raise
    mapping[exp_name] = module.TrainConfig


CONFIG_MAPPING = {}

_register(CONFIG_MAPPING, "pick_place_banana", "pick_place_banana")

