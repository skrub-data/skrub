from functools import wraps


class with_setup:
    """
    Deprecated.
    Might be useful though.
    """
    def __init__(self, setup, teardown):
        self.setup = setup
        self.teardown = teardown

    def __call__(self, f):
        @wraps(f)
        def func(*args, **kwargs):
            self.setup()
            try:
                f(*args, **kwargs)
            finally:
                self.teardown()

        return func
