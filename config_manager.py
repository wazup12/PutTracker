import configparser
import os


class ConfigManager:
    """Handles reading and writing configuration from config.ini."""

    def __init__(self, config_path="config.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.read_config()

    def read_config(self):
        self.config.read(self.config_path)
        if "DEBUG" not in self.config["DEFAULT"]:
            self.config["DEFAULT"]["DEBUG"] = "false"
            self.write_config()
        return self.config

    def write_config(self):
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)

    def get(self, section, key, default=None):
        return self.config[section].get(key, default)

    def set(self, section, key, value):
        self.config[section][key] = value
        self.write_config()

    def get_file_directory(self):
        return os.path.expanduser(self.get("DEFAULT", "file_directory", "."))

    def set_file_directory(self, new_path):
        self.set("DEFAULT", "file_directory", new_path)
