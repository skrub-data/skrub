def list_intersect(values, allowed):
    return [v for v in values if v in set(allowed)]


def list_difference(values, excluded):
    return [v for v in values if v not in set(excluded)]
