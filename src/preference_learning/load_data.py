import pandas as pd
import dotenv
import pathlib
import os

from preference_learning import NumpyDataset


def load_dataframe(mode: str = "raw") -> pd.DataFrame:
    """Returns a pandas dataframe with the car evaluation dataset."""

    modes = ["raw", "processed"]
    if mode not in modes:
        raise ValueError(f"Mode must be one of {modes}.")

    # Load data
    dotenv.load_dotenv()
    project_path = pathlib.Path(os.getenv("PROJECT_PATH"))
    data_path = project_path / "data" / f"car_evaluation_{mode}.csv"
    columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    df = pd.read_csv(data_path, names=columns)

    return df


def load_dataset(mode: str = "raw") -> NumpyDataset:
    """Returns a torch dataset with the car evaluation dataset."""

    df = load_dataframe(mode=mode)
    X = df.drop(columns=["class"]).values
    y = df["class"].values

    return NumpyDataset(X, y)


if __name__ == "__main__":
    print(load_dataframe(mode="processed").head())
    columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    print("N classes:")
    for col in columns:
        print(f"{col}: {len(load_dataframe(mode='processed')[col].unique())}")