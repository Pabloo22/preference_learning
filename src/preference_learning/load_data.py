import pandas as pd
import dotenv
import pathlib
import os
from typing import Tuple, Union

from preference_learning import NumpyDataset


TupleOfDataFrames = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]


def load_dataframe(mode: str = "raw") -> Union[pd.DataFrame, TupleOfDataFrames]:
    """Returns a pandas dataframe with the car evaluation dataset."""

    modes = ["raw", "processed", "split"]
    if mode not in modes:
        raise ValueError(f"Mode must be one of {modes}.")

    # Load data
    dotenv.load_dotenv()
    project_path = pathlib.Path(os.getenv("PROJECT_PATH"))

    if mode != "split":
        data_path = project_path / "data" / f"car_evaluation_{mode}.csv"
        columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

        df = pd.read_csv(data_path, names=columns)

        return df

    X_train_path = project_path / "data" / "X_train.csv"
    X_test_path = project_path / "data" / "X_test.csv"
    y_train_path = project_path / "data" / "y_train.csv"
    y_test_path = project_path / "data" / "y_test.csv"

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)

    return X_train, X_test, y_train, y_test


def load_dataset(mode: str = "raw") -> Union[NumpyDataset, Tuple[NumpyDataset, NumpyDataset]]:
    """Returns a torch dataset with the car evaluation dataset."""

    if mode == "split":
        X_train, X_test, y_train, y_test = load_dataframe(mode=mode)
        return NumpyDataset(X_train.values, y_train.values), NumpyDataset(X_test.values, y_test.values)

    df = load_dataframe(mode=mode)
    X = df.drop(columns=["class"]).values
    y = df["class"].values

    return NumpyDataset(X, y)
