from dirty_cat.utils import LRUDict


def test_lrudict():
    dict_ = LRUDict(10)

    for x in range(15):
        dict_[x] = 'filled'+str(x)

    for x in range(5, 15):
        assert x in dict_
        assert dict_[x] == 'filled'+str(x)

    for x in range(5):
        assert (x not in dict_)
