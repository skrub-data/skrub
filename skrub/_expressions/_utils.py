FITTED_PREDICTOR_METHODS = ("predict", "predict_proba", "decision_function", "score")
FITTED_ESTIMATOR_METHODS = FITTED_PREDICTOR_METHODS + ("transform",)
X_NAME = "_skrub_X"
Y_NAME = "_skrub_y"


def simple_repr(expr, open_tag="", close_tag=""):
    text = repr(expr).splitlines()[0].removeprefix("<").removesuffix(">")
    start, sep, rest = text.partition(" ")
    return f"{open_tag}{start.upper()}{close_tag}{sep}{rest}"


def attribute_error(obj, name, comment=None):
    msg = f"{obj.__class__.__name__!r} object has no attribute {name!r}"
    if comment:
        msg = f"{msg}. {comment}"
    raise AttributeError(msg)


class _CloudPickle:
    def __getstate__(self):
        from joblib.externals import cloudpickle

        try:
            state = dict(super().__getstate__())
        except AttributeError:
            # before python 3.11
            state = self.__dict__.copy()
        for k in self._cloudpickle_attributes:
            # TODO warn if cloudpickle has to pickle by value
            state[k] = cloudpickle.dumps(state[k])
        return state

    def __setstate__(self, state):
        from joblib.externals import cloudpickle

        for k in self._cloudpickle_attributes:
            state[k] = cloudpickle.loads(state[k])
        self.__dict__ = state
