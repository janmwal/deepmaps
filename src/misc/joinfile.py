import os

def joinfile(*args):
    return os.path.normpath(os.path.join(*args))