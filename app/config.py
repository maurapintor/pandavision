import dotenv
import os
import pathlib

current_dir = pathlib.Path(__file__)

try:
    project_dir = [p for p in current_dir.parents if p.parts[-1] == 'pandavision'][0]
except:
    project_dir = "."
config_file_path = dotenv.dotenv_values(dotenv_path=os.path.join(project_dir, ".env"))


class Config(dict):

    def __getattr__(self, name):
        return self[name] if not isinstance(self[name], dict) \
            else Config(self[name])


config = Config(config_file_path)
config['PROJECT_ROOT'] = project_dir
import os
SECRET_KEY = os.urandom(32)
config['SECRET_KEY'] = SECRET_KEY