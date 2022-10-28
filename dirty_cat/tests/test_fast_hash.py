import random

from dirty_cat._fast_hash import ngram_min_hash


def generate_data(n_samples):
    MAX_LIMIT = 255  # extended ASCII Character set
    i = 0
    str_list = []
    for i in range(n_samples):
        random_string = "category "
        for _ in range(100):
            random_integer = random.randint(0, MAX_LIMIT)
            random_string += chr(random_integer)
            if random_integer < 50:
                random_string += "  "
        i += 1
        str_list += [random_string]
    return str_list


def test_fast_hash():
    data = generate_data(100)
    a = data[0]

    min_hash = ngram_min_hash(a, seed=0)
    min_hash2 = ngram_min_hash(a, seed=0)
    assert min_hash == min_hash2

    list_min_hash = [ngram_min_hash(a, seed=seed) for seed in range(50)]
    assert len(set(list_min_hash)) > 45, "Too many hash collisions"

    min_hash4 = ngram_min_hash(a, seed=0, return_minmax=True)
    assert len(min_hash4) == 2
