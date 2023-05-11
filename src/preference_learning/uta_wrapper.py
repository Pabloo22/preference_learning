import numpy as np
import pandas as pd
import torch

from preference_learning import create_data_loader, train, accuracy, set_seed, Uta, NormLayer
import functools


class UtaWrapper:
    """Scikit-learn wrapper for the Uta Model"""

    def __init__(self,
                 criteria_nr=6,
                 hidden_nr=3,
                 epochs=472,
                 learning_rate=0.0420,
                 seed=123):
        self.criteria_nr = criteria_nr
        self.hidden_nr = hidden_nr
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.model_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        set_seed(self.seed)

        # Create DataLoaders
        train_dataloader = create_data_loader(
            X.to_numpy().reshape((-1, 1, self.criteria_nr)),
            y.to_numpy())

        # Create Model
        uta = Uta(criteria_nr=self.criteria_nr, hidden_nr=self.hidden_nr)
        model = NormLayer(uta, self.criteria_nr)

        acc_fn = functools.partial(accuracy, threshold=0)

        # Train Model
        model = train(
                    train_dataloader=train_dataloader,
                    test_dataloader=None,
                    model=model,
                    lr=self.learning_rate,
                    epoch_nr=self.epochs,
                    accuracy_fn=acc_fn,
                    save_model=False,
                    use_test_set=False,
                    verbose=False,
                    return_best_model=True,
                    )
        self.model_ = model

    def load_model(self, path):
        uta = Uta(criteria_nr=self.criteria_nr, hidden_nr=self.hidden_nr)
        model = NormLayer(uta, self.criteria_nr)
        saved_model = torch.load(path)
        model.load_state_dict(saved_model["model_state_dict"])
        self.model_ = model

    def save_model(self, path):
        torch.save({
            "model_state_dict": self.model_.state_dict(),
            }, path)

    def predict(self, X: pd.DataFrame):
        if self.model_ is None:
            raise RuntimeError("The model must be trained before making predictions.")

        X_tensor = torch.tensor(X.to_numpy().reshape((-1, 1, self.criteria_nr)), dtype=torch.float32)
        with torch.no_grad():
            preds = self.model_(X_tensor)
            preds_binary = (preds > 0).squeeze().numpy().astype(int)

        return preds_binary.reshape((-1, 1))

    def predict_proba(self, X: pd.DataFrame):
        if self.model_ is None:
            raise RuntimeError("The model must be trained before making predictions.")

        X_tensor = torch.tensor(X.to_numpy().reshape((-1, 1, self.criteria_nr)), dtype=torch.float32)
        with torch.no_grad():
            preds = self.model_(X_tensor)
            proba = torch.sigmoid(preds).squeeze().numpy()

        # Add a column for the negative class
        proba_neg = 1 - proba
        proba = np.concatenate((proba_neg.reshape((-1, 1)), proba.reshape((-1, 1))), axis=1)

        return proba

    def __call__(self, X: pd.DataFrame):
        return self.predict(X)
