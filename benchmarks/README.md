# skrub benchmarks

## Objectives

This folder contains benchmarks used by the skrub maintainers to:
- Experiment on new algorithms
- Validate decisions based on empirical evidence
- Fine-tune (hyper)parameters in the library

These benchmarks do not aim at replacing the tests within skrub.

## Implementing a benchmark

A mini-framework consisting of a few functions is made available under `utils`.

Check out other benchmarks to see how they are used.

## Launching a benchmark

> Launching a benchmark is usually something you don't want to do as a user.
  Benchmarks are long and expensive to run, and are here mainly for reproducibility.

Each one implements a standard command-line interface with the two commands
``--run`` and ``--plot``.

For instance, for running a benchmark, we'll usually use

```bash
python tablevectorizer_tuning.py --run
```

Package requirements required to launch benchmarks are listed in the project's
setup, thus can be installed with

```bash
pip install -e .[benchmarks]
```

It has been reported that Python >=3.9 is required.

### Analyzing results

The results of the benchmarks ran by maintainers are pushed in the `results/` folder.

As mentioned earlier, benchmarks implement a ``--plot`` parameter used
to display the results visually. Using ``--plot`` without ``--run`` allows you to plot the results without re-running the benchmark.
