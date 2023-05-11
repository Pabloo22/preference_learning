from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import pathlib
import dotenv
import os
import torch
import skorch

from preference_learning import load_dataframe, UtaWrapper

def main():
    dotenv.load_dotenv()
    X_train, X_test, y_train, y_test = load_dataframe(mode="split")
    model = torch.load(pathlib.Path(os.getenv("PROJECT_PATH")) / "models" / "ann_utadis.pt")
    model = skorch.NeuralNetClassifier(model)
    model.initialize()
    explainer = ClassifierExplainer(model, X_test, y_test)
    db = ExplainerDashboard(explainer, title="Car Evaluation Dashboard")
    db.run()


if __name__ == "__main__":
    main()
