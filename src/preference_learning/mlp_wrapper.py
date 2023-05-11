import pandas as pd
import torch

from preference_learning import create_data_loader, train, set_seed


class MLPWrapper:
    """Scikit-learn wrapper for the simple MLP model"""

    def __init__(self, input_dim=6, hidden_dim=3, epochs=200, learning_rate=0.01, seed=123):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        set_seed(self.seed)

        # Create DataLoaders
        train_dataloader = create_data_loader(
            X.to_numpy(),
            y.to_numpy().reshape(-1,))

        # Create Model
        model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
            torch.nn.Sigmoid(),
        )
        loss_fn = torch.nn.BCELoss()

        # Train Model
        model = train(
                    train_dataloader=train_dataloader,
                    test_dataloader=None,
                    model=model,
                    lr=self.learning_rate,
                    epoch_nr=self.epochs,
                    loss_fn=loss_fn,
                    save_model=False,
                    use_test_set=False,
                    verbose=False,
                    return_best_model=True,
                    )
        self.model = model

    def load_model(self, path):
        saved_model = torch.load(path)

        # Create a new instance of the model architecture
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
            torch.nn.Sigmoid(),
        )

        # Copy the weights from the saved model to the new instance
        self.model.load_state_dict(saved_model["model_state_dict"])

    def predict(self, X: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("The model must be trained before making predictions.")

        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X_tensor)
            preds_binary = (preds > 0.5).squeeze().numpy().astype(int)

        return preds_binary

    def predict_proba(self, X: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("The model must be trained before making predictions.")

        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
        with torch.no_grad():
            proba = self.model(X_tensor).squeeze().numpy()

        return proba
