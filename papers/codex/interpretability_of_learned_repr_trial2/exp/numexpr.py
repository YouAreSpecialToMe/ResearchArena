__version__ = "0.0.0"

MAX_THREADS = 1
use_vml = False
__BLOCK_SIZE1__ = 1024


def evaluate(*args, **kwargs):
    raise NotImplementedError("Local numexpr stub: vectorized evaluation is disabled in this workspace.")


def re_evaluate(*args, **kwargs):
    raise NotImplementedError("Local numexpr stub: vectorized evaluation is disabled in this workspace.")


def set_num_threads(n):
    return 1


def detect_number_of_cores():
    return 1


def get_vml_version():
    return None
