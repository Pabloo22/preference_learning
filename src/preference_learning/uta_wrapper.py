import pandas as pd
import torch

from preference_learning import create_data_loader, train, accuracy, set_seed, Uta, NormLayer
import functools


class UtaWrapper:
    """Scikit-learn wrapper for the Uta Model"""

    def __init__(self, criteria_nr=6, hidden_nr=3, epochs=472, learning_rate=0.0420, seed=123):
        self.criteria_nr = criteria_nr
        self.hidden_nr = hidden_nr
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = None

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
        self.model = model

    def load_model(self, path):
        uta = Uta(criteria_nr=self.criteria_nr, hidden_nr=self.hidden_nr)
        model = NormLayer(uta, self.criteria_nr)
        model.load_state_dict(torch.load(path)["model_state_dict"])
        self.model = model

    def predict(self, X: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("The model must be trained before making predictions.")

        X_tensor = torch.tensor(X.to_numpy().reshape((-1, 1, self.criteria_nr)), dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X_tensor)
            preds_binary = (preds > 0).squeeze().numpy().astype(int)

        return preds_binary

    def predict_proba(self, X: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("The model must be trained before making predictions.")

        X_tensor = torch.tensor(X.to_numpy().reshape((-1, 1, self.criteria_nr)), dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X_tensor)
            proba = torch.sigmoid(preds).squeeze().numpy()

        return proba

