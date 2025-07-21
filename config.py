import argparse
import yaml

class ConfigDict(dict):
    """A dictionary that allows for attribute-style access."""
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value
    def __getstate__(self):
        return dict(self)
    def __setstate__(self, state):
        self.update(state)

def _convert_scientific_notation(value):
    """Convert string scientific notation to float if applicable."""
    if isinstance(value, str):
        try:
            if 'e-' in value.lower() or 'e+' in value.lower():
                return float(value)
        except ValueError:
            pass
    return value

def get_config():
    """
    Loads a configuration file and merges it with command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")

    args, unknown = parser.parse_known_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        config[key] = _convert_scientific_notation(value)

    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", type=lambda x: (str(x).lower() == 'true'), default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    args = parser.parse_args()
    
    config = ConfigDict(vars(args))

    return config 