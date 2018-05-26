import pathlib


def setup(name):
    pathlib.Path(name).mkdir(parents=True, exist_ok=True)
    print("Start Processing: " + name)
    return name
