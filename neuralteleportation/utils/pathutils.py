import os
from os.path import join as pjoin


def get_nonexistent_path(fname_path, mkdir=True):
    """
    Get the path to a filename which does not exist by incrementing path.

    Examples
    --------
    get_nonexistant_path('/etc/issue')
    '/etc/issue_1'

    Args:
        fname_path (str): string, path to increment

    Returns:
        non existent file path
    """
    if not os.path.exists(fname_path):
        if mkdir:
            os.makedirs(fname_path)
        return fname_path
    filename, file_extension = os.path.splitext(fname_path)
    i = 1
    new_fname = "{}_{}{}".format(filename, i, file_extension)
    while os.path.exists(new_fname):
        i += 1
        new_fname = "{}_{}{}".format(filename, i, file_extension)
    if mkdir:
        os.mkdir(new_fname)
    return new_fname