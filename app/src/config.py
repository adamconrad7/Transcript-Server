import yaml

def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
