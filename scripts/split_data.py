from sklearn.model_selection import train_test_split
import dotenv
import pathlib
import os

from preference_learning import load_dataframe, set_seed


def main():

    set_seed(123)
    df = load_dataframe(mode="processed")
    X = df.drop(columns=["class"])

    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the dataframes
    dotenv.load_dotenv()
    project_path = pathlib.Path(os.getenv("PROJECT_PATH"))
    path = project_path / "data"
    X_train.to_csv(path / "X_train.csv", index=False)
    X_test.to_csv(path / "X_test.csv", index=False)
    y_train.to_csv(path / "y_train.csv", index=False)
    y_test.to_csv(path / "y_test.csv", index=False)


if __name__ == "__main__":
    main()
