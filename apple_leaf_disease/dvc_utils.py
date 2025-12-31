import subprocess
from pathlib import Path


def pull_data_if_missing(raw_dir: Path, train_csv_name: str = 'train.csv') -> None:
    train_csv_path = raw_dir / train_csv_name

    if train_csv_path.exists():
        return

    subprocess.run(['dvc', 'pull'], check=True)

    if not train_csv_path.exists():
        raise FileNotFoundError(
            f'{train_csv_path} not found after `dvc pull`. Check dvc-tracked paths.'
        )
