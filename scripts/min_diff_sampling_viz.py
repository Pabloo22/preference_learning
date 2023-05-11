import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dotenv
import pathlib
import os

from preference_learning import UtaWrapper, MLPWrapper, load_dataframe


dotenv.load_dotenv()
project_path = pathlib.Path(os.getenv("PROJECT_PATH"))


def plot_feature_importance(model, X_train: pd.DataFrame, prefix: str = "mlp"):

    # Create a grid of 50 evenly spaced points between 0 and 1
    grid = np.linspace(0, 1, 50)

    # Iterate through each feature (criteria)
    criteria = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    for j, c in enumerate(criteria):
        y_pred = np.zeros(len(grid))

        # For each value in the grid, perform the following steps:
        for i, val in enumerate(grid):
            X_temp = X_train.values.copy()
            X_temp[:, j] = val

            y_pred[i] = np.average(model.predict_proba(pd.DataFrame(X_temp))[:, 1])

        # Create a new Matplotlib figure and axis
        fig, ax = plt.subplots()

        # Plot the grid values on the x-axis and the average prediction scores on the y-axis
        ax.plot(grid, y_pred, label='average score')

        # Set the x-axis limits to 0 and 1
        ax.set_xlim(0, 1)

        # Label the x-axis with the current feature (criteria) number
        ax.set_xlabel(f'Feature {c}')

        # Add a legend to the plot
        ax.legend()

        # Save the figure
        fig.savefig(project_path / "plots" / f"{prefix}_feature_importance_{c}.png", dpi=300)

        # Close the figure
        plt.close(fig)


def mlp():
    X_train, *_ = load_dataframe(mode="split")
    model = MLPWrapper()
    path = project_path / "models" / "ann.pt"
    model.load_model(path)
    plot_feature_importance(model, X_train)


def uta():
    X_train, *_ = load_dataframe(mode="split", all_gain=True)
    model = UtaWrapper()
    path = project_path / "models" / "ann_utadis.pt"
    model.load_model(path)
    plot_feature_importance(model, X_train, prefix="uta")


if __name__ == "__main__":
    mlp()
    uta()
