from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import pathlib
import dotenv
import os

from preference_learning import load_dataframe, MLPWrapper


def main():
    dotenv.load_dotenv()
    X_train, X_test, y_train, y_test = load_dataframe(mode="split")
    model = MLPWrapper()
    path = pathlib.Path(os.getenv("PROJECT_PATH")) / "models" / "ann.pt"
    model.load_model(path)
    explainer = ClassifierExplainer(model, X_test, y_test)
    db = ExplainerDashboard(explainer, title="MLP - Car Evaluation Dashboard")
    db.run()


if __name__ == "__main__":
    main()
