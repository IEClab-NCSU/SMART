import os
import errno

def safe_open_w(path):
    # Open "path" for writing, creating any parent directories as needed.
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exc: # Python >2.5 (Guard against race condition)
        if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(path)):
            pass
        else: raise
    return open(path, 'w')

def safe_open_wb(path):
    # Open "path" for writing, creating any parent directories as needed.
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exc: # Python >2.5 (Guard against race condition)
        if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(path)):
            pass
        else: raise
    return open(path, 'wb')