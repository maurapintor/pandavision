import dotenv
import os
import pathlib

current_dir = pathlib.Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1] == 'pandavision'][0]
config = dotenv.dotenv_values(dotenv_path=os.path.join(project_dir, ".env"))


class Config(dict):

    def __getattr__(self, name):
        return self[name] if not isinstance(self[name], dict) \
            else Config(self[name])

