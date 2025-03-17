import skrub
from skrub._expressions import _evaluation


def test_caching():
    a = skrub.var("a")
    b = a + a
    c = b + a
    d = c + a

    def check_cache():
        # before the first node is evaluated all caches are empty
        assert a._skrub_impl.results == {}
        assert b._skrub_impl.results == {}
        assert c._skrub_impl.results == {}
        assert d._skrub_impl.results == {}
        yield
        # a has been computed
        assert a._skrub_impl.results == {"fit_transform": 10}
        assert b._skrub_impl.results == {}
        assert c._skrub_impl.results == {}
        assert d._skrub_impl.results == {}
        yield
        # b has been computed, a is still needed so both are in the cache
        assert a._skrub_impl.results == {"fit_transform": 10}
        assert b._skrub_impl.results == {"fit_transform": 20}
        assert c._skrub_impl.results == {}
        assert d._skrub_impl.results == {}
        yield
        # c has been computed, b is not needed any more, a is still needed for d
        assert a._skrub_impl.results == {"fit_transform": 10}
        assert b._skrub_impl.results == {}
        assert c._skrub_impl.results == {"fit_transform": 30}
        assert d._skrub_impl.results == {}
        yield
        # d has been computed, a and c are not needed anymore
        assert a._skrub_impl.results == {}
        assert b._skrub_impl.results == {}
        assert c._skrub_impl.results == {}
        assert d._skrub_impl.results == {"fit_transform": 40}
        yield

    check = check_cache()
    next(check)
    _evaluation.evaluate(
        d,
        mode="fit_transform",
        environment={"a": 10},
        clear=True,
        callbacks=((lambda e, r: next(check)),),
    )

    # the check generator is exhausted (we reached the last yield)
    assert next(check, "finished") == "finished"

    # and the last remaining result has been cleared from the cache as well
    assert a._skrub_impl.results == {}
    assert b._skrub_impl.results == {}
    assert c._skrub_impl.results == {}
    assert d._skrub_impl.results == {}
