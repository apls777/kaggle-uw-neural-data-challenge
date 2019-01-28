import os


def root_dir(path=''):
    # return the path if it's already absolute
    if path and (is_s3_path(path) or os.path.isabs(path)):
        return path

    res_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if path:
        res_path = os.path.join(res_path, path)

    return res_path


def is_s3_path(path):
    return path.startswith('s3://')


def fix_s3_separators(path):
    """Fixes S3 path if it was affected by Windows path separators."""
    if is_s3_path(path):
        path = path.replace('\\', '/')

    return path
