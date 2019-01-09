import os


def root_dir(path=''):
    res_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if path:
        res_path = os.path.join(res_path, path)

    return res_path
