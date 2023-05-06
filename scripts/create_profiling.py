import dotenv
import pandas as pd
import pathlib
import os
from ydata_profiling import ProfileReport

from preference_learning import load_dataframe


def create_profiling(df: pd.DataFrame, output_path: pathlib.Path):
    """Create profiling report for the given dataframe.

    Args:
        df (pd.DataFrame): Dataframe to create profiling for.
        output_path (pathlib.Path): Path to save the profiling report.
    """
    profile = ProfileReport(df, title="Car Evaluation Profiling Report")
    profile.to_file(output_file=output_path)


def main():
    dotenv.load_dotenv()
    project_path = pathlib.Path(os.getenv("PROJECT_PATH"))
    # Create folder if it doesn't exist
    os.makedirs(project_path / "reports", exist_ok=True)
    output_path = project_path / "reports" / "car_evaluation_profiling.html"

    df = load_dataframe(mode="processed")

    create_profiling(df, output_path)


if __name__ == "__main__":
    main()
