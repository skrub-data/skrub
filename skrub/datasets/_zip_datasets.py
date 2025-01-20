import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path

from skrub import datasets


def create_archive(
    all_datasets_dir, all_archives_dir, dataset_name, dataframes, metadata
):
    print(dataset_name)
    dataset_dir = all_datasets_dir / dataset_name
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata), "utf-8")
    for stem, df in dataframes.items():
        csv_path = dataset_dir / f"{stem}.csv"
        df.to_csv(csv_path, index=False)
    archive_path = all_archives_dir / dataset_name
    result = shutil.make_archive(
        archive_path,
        "zip",
        root_dir=all_datasets_dir,
        base_dir=dataset_name,
    )
    result = Path(result)
    checksum = hashlib.sha256(result.read_bytes()).hexdigest()
    return checksum


def load_simple_dataset(fetcher):
    dataset = fetcher()
    df = dataset.X
    df[dataset.target] = dataset.y
    name = fetcher.__name__.removeprefix("fetch_")
    return (
        name,
        {name: df},
        {
            "name": dataset.name,
            "description": dataset.description,
            "source": dataset.source,
            "target": dataset.target,
        },
    )


def iter_datasets():
    simple_fetchers = {f for f in datasets.__all__ if f.startswith("fetch_")} - {
        "fetch_world_bank_indicator",
        "fetch_figshare",
        "fetch_credit_fraud",
        "fetch_ken_embeddings",
        "fetch_ken_table_aliases",
        "fetch_ken_types",
    }
    for fetcher in sorted(simple_fetchers):
        yield load_simple_dataset(getattr(datasets, fetcher))


def make_skrub_datasets():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="where to store the output. a subdirectory containing all the archives will be created",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(args.output_dir).resolve()

    root_dir = (
        output_dir / f"skrub_datasets_{datetime.datetime.now():%Y-%m-%dT%H-%M%S}"
    )
    root_dir.mkdir(parents=True)
    all_datasets_dir = root_dir / "datasets"
    all_datasets_dir.mkdir()
    all_archives_dir = root_dir / "archives"
    all_archives_dir.mkdir()

    print(f"saving output in {root_dir}")

    checksums = {}
    for dataset_name, dataframes, metadata in iter_datasets():
        checksums[dataset_name] = create_archive(
            all_datasets_dir, all_archives_dir, dataset_name, dataframes, metadata
        )

    (all_archives_dir / "checksums.json").write_text(json.dumps(checksums), "utf-8")
    print(f"archive files saved in {all_archives_dir}")


if __name__ == "__main__":
    make_skrub_datasets()