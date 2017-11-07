import json
DEFAULT_CONFIG_PATH = './config/default.json'


def load_config(config_path):
    config_file_path = config_path if config_path else DEFAULT_CONFIG_PATH
    if not config_path:
        print("No config file specified...using default config!"
              "To use a different config please pass an argument"
              "(e.g python a3c.py configs/your_config.yaml)")

    with open(config_file_path, 'r') as f:
        config = json.load(f)

    return config