import dotenv
import os
import pathlib

from preference_learning import load_dataframe


def create_processed_data():
    df = load_dataframe(mode="raw")
    # binarize class column
    df["class"] = (df["class"] > 2).astype(int)

    # save processed data
    dotenv.load_dotenv()
    project_path = pathlib.Path(os.getenv("PROJECT_PATH"))
    data_path = project_path / "data" / "car_evaluation_processed.csv"
    df.to_csv(data_path, index=False, header=False)


if __name__ == "__main__":
    create_processed_data()
