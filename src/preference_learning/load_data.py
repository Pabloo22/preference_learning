import pandas as pd
import dotenv
import pathlib
import os


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


if __name__ == "__main__":
    print(load_dataframe(mode="processed").head())