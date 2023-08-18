# skrub benchmarks

## Objectives

This folder contains benchmarks used by the skrub maintainers to:
- Experiment on new algorithms
- Validate decisions based on empirical evidence
- Tune (hyper)parameters in the library

These benchmarks do not aim at replacing the tests within skrub.

## Implementing a benchmark

A mini-framework consisting of a few functions is made available under `utils`.

Check out other benchmarks to see how they are used.

## Launching a benchmark

> Launching a benchmark is usually something you don't want to do as a user.
  Benchmarks are long and expensive to run. Their code is provided for reproducibility.

Each one implements a standard command-line interface with the at least the two
commands ``--run`` and ``--plot``.

Although, before launching, you should make sure the environment is properly setup.
First, install the required packages -- we recommend installing the latest versions
for everything (skip `--upgrade` if you don't want to):

```bash
pip install -e --upgrade .[benchmarks]
```

It has also been reported that Python >=3.9 is required.

Then, if you're trying to reproduce the results of a benchmark, check the file's
docstring to see if it requires any additional setup.
Usually, you will find a date, which might be relevant, and sometimes, a commit
hash. You can use it to checkout the code at the time the benchmark was run:

```bash
git checkout <commit_hash>
```

Finally, you can launch the benchmark with the ``--run`` command:

```bash
python bench_tablevectorizer_tuning.py --run
```

### Analyzing results

The results of the benchmarks ran by maintainers are pushed in the `results/`
folder in a `parquet` format.

As mentioned earlier, benchmarks implement a ``--plot`` option used to display
the results visually. Using ``--plot`` without ``--run`` allows you to plot
the saved results without re-running the benchmark.
