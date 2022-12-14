# dirty_cat benchmarks

## Objectives

This folder contains benchmarks used by the dirty_cat maintainers to:
- Experiment on new algorithms
- Validate decisions based on empirical evidence
- Fine-tune (hyper)parameters in the library

These benchmarks do not aim at replacing the tests within dirty_cat.

## Implementing a benchmark

A mini-framework consisting of a few functions is made available under `utils`.

Check out other benchmarks to see how they are used.

If you're trying to benchmark an experimental feature,
and as benchmark files are "stand-alone" (they use `import dirty_cat`),
you'll have to make your copy of dirty_cat available to your Python executable.

To do that, the recommended way is to create a development environment as
explained in the project's contributing guidelines, and, if you're on linux,
add the path to your dirty_cat local repository in the Python path with the command

```bash
export PYTHONPATH="$PYTHONPATH:</path/to/repository/>"
```

Now, when launching `python` then `import dirty_cat`, it should import
your local copy containing your experimental features.

## Launching a benchmark

> Launching a benchmark is usually something you don't want to do as a user.
  Benchmarks are long and expensive to run, and are here mainly for reproducibility.

Each one implements a standard command-line interface with the two commands
``--run`` and ``--plot``.

For instance, for running a benchmark, we'll usually use

```bash
python supervectorizer_tuning.py --run
```

### Analyzing results

The results of the benchmarks ran by maintainers are pushed in the `results/` folder.

As mentioned earlier, benchmarks implement a ``--plot`` parameter used
to display the results visually.
