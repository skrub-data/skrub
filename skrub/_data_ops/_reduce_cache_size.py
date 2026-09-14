import argparse

import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=str)
    parser.add_argument("target_size", type=str)
    args = parser.parse_args()

    cache_dir = args.cache_dir
    size = args.target_size
    try:
        size = int(size)
    except ValueError:
        pass

    # Always pass a string to Memory because it behaves differently if
    # we pass a Path (adds joblib/ for strings but not for Paths)
    # https://github.com/joblib/joblib/issues/1684
    memory = joblib.Memory(str(cache_dir))

    memory.reduce_size(bytes_limit=size)


if __name__ == "__main__":
    main()
