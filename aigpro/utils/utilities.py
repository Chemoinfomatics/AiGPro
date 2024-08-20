import os
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import StratifiedKFold

console = Console()


def basedir_to_file(base_dir) -> Tuple[list, dict]:
    """Converts the base directory to a list of directories.

    Args:
        base_dir (_type_): _description_

    Returns:
        Tuple[list, dict]: _description_
    """
    all_dirs: list[str] = os.listdir(base_dir)
    struct_dirs: list = []
    no_structured_dirs: list = []
    complete_dataset: dict = {}
    for i in all_dirs:
        x: Path = Path(f"{base_dir}/{i}")
        sample_protein: str = f"{x}/{x.name}_protein.pdb"
        sample_ligand: str = f"{x}/{x.name}_ligand.sdf"

        sample_protein, sample_ligand = os.path.abspath(sample_protein), os.path.abspath(sample_ligand)
        if (os.path.exists(sample_protein)) and (os.path.exists(sample_ligand)):
            struct_dirs.append(sample_protein)
            complete_dataset[x.name] = {
                "protein": sample_protein,
                "ligand": sample_ligand,
            }

        else:
            no_structured_dirs.append(sample_protein)
            print(f"File not found {x.name}")
    return struct_dirs, complete_dataset


def index_to_df(file) -> pd.DataFrame:
    """Converts the index file to a dataframe.

    Args:
        file (_type_): _description_

    Returns:
        pd.DataFrame: _description_
    """
    with open(file, "r") as f:
        pdbbind_index: list[str] = f.readlines()
    refined_dict: dict = {}

    for line in pdbbind_index:
        line: str = line.strip()
        if not line.startswith("#"):
            # pdbbind_index = json.loads(line)
            pdb_id = line.split()[0]
            resolution = line.split()[1]
            year = line.split()[2]
            logkd_ki = line.split()[3]
            exp = line.split()[4]
            exp_name, exp_value = exp.split("=")
            exp_value = float(exp_value[:-2])
            ligand_name = line.split()[-1].replace("(", "").replace(")", "").strip()
            refined_dict[pdb_id] = {
                "resolution": resolution,
                "year": year,
                "exp": exp,
                "ligand_name": ligand_name,
                "exp_name": exp_name,
                "exp_value": exp_value,
                "logkd_ki": logkd_ki,
            }
    return pd.DataFrame.from_dict(refined_dict, orient="index")


def plot_data(lengths, error=None, posy=850):
    """Plots the data.

    Args:
        lengths (_type_): _description_
        error (_type_, optional): _description_. Defaults to None.
        posy (int, optional): _description_. Defaults to 850.
    """
    if not error:
        error = []
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=100)
    ax.axvline(max(lengths), color="red")
    ax.text(max(lengths), 0, f"max length: {max(lengths)}", color="red")
    ax.axvline(min(lengths), color="green")
    ax.text(min(lengths), posy, f"min length: {min(lengths)}", color="green")
    plt.title(f"Distribution of protein sequence lengths from core PDBbind: {len(lengths)-len(error)}")
    plt.show()


def get_pad_array(cmap, pad_size=(600, 600), constant=-1):
    """Pads the array.

    Args:
        cmap (_type_): _description_
        pad_size (tuple, optional): _description_. Defaults to (600, 600).
        constant (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """
    assert len(pad_size) == 2, "Padding size bust be 2D"
    try:
        if cmap.shape[0] > pad_size[0]:
            cmap = cmap[: pad_size[0], :]

        if cmap.shape[1] > pad_size[1]:
            cmap = cmap[:, : pad_size[1]]
        cmap = np.pad(
            cmap,
            ((0, pad_size[0] - cmap.shape[0]), (0, pad_size[1] - cmap.shape[1])),
            "constant",
            constant_values=constant,
        )
        return cmap
    except Exception as e:
        print(e)
        return None


def parse_yaml(filename):
    """Parses the yaml file.

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(filename, "r") as stream:
        try:
            console.print("Cofiguration file loaded")
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)


def config_to_args(config, args):
    """Converts the config to args.

    Args:
        config (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    # args = argparse.Namespace()
    for k, v in config.items():
        setattr(args, k, v)
    return args


def add_fold(df, K=5, column_name="fold", seed=42, target_col="target"):
    """Adds the fold to the dataframe.

    Args:
        df (_type_): _description_
        K (int, optional): _description_. Defaults to 5.
        column_name (str, optional): _description_. Defaults to "fold".
        seed (int, optional): _description_. Defaults to 42.
        target_col (str, optional): _description_. Defaults to "target".

    Returns:
        _type_: _description_
    """
    target_col = "target" if not column_name else column_name
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold_number, (train_index, test_index) in enumerate(skf.split(df, df[target_col])):
        df.loc[test_index, "fold"] = fold_number
    return df


def load_dataframe(filename, header=0, skiprows=None, sep=",", index_col=None):
    """Loads the dataframe.

    Args:
        filename (_type_): _description_
        header (int, optional): _description_. Defaults to 0.
        skiprows (_type_, optional): _description_. Defaults to None.
        sep (str, optional): _description_. Defaults to ",".
        index_col (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    filename = Path(filename)

    match filename.suffix:
        case ".csv":
            df = pd.read_csv(filename, header=header, skiprows=skiprows, sep=sep, index_col=index_col)
        case ".tsv":
            df = pd.read_csv(filename, sep="\t", header=header, skiprows=skiprows, index_col=index_col)
        case ".xlsx":
            df = pd.read_excel(filename)
        case ".json":
            df = pd.read_json(filename)
        case ".pkl":
            df = pd.read_pickle(filename)
        case ".parquet":
            df = pd.read_parquet(filename, engine="pyarrow")
        case _:
            raise ValueError(f"Invalid file format `{filename.suffix}` for {filename}")
    return df.sample(frac=1).reset_index(drop=True)


def print_metric_table(metric_title, additional_metrics):
    """Adds the metric table."""
    table = Table(title=f"{metric_title}")

    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta", justify="left", no_wrap=True)

    for metric, value in additional_metrics.items():
        table.add_row(metric, str(value))

    console = Console()
    console.print(table)
