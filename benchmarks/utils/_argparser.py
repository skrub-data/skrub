from argparse import ArgumentParser

# Inherit this parser in your file, like so:
# >>> _parser = ArgumentParser(parents=[default_parser])

default_parser = ArgumentParser(add_help=False)

default_parser.add_argument(
    "--run",
    help="Runs the benchmark.",
    action="store_true",
)
default_parser.add_argument(
    "--plot",
    help=(
        "Plots the results. If '--run' is specified, plots those. "
        "Otherwise, searches for result files in the results' directory. "
        "If it finds none, exits. If it finds only one, displays it. "
        "If it finds multiple, prompts the user to choose one. "
    ),
    action="store_true",
)
