from sklearn.model_selection import train_test_split
import dotenv
import pathlib
import torch
import os

from preference_learning import load_dataframe, create_data_loader, train, evaluate_model, set_seed


def main():

    set_seed(123)
    df = load_dataframe(mode="processed")
    X = df.drop(columns=["class"]).to_numpy()  # X.shape = (1728, 6)

    y = df["class"].to_numpy()  # y.shape = (1728,)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    train_dataloader = create_data_loader(X_train, y_train)
    val_dataloader = create_data_loader(X_val, y_val)
    test_dataloader = create_data_loader(X_test, y_test)

    dotenv.load_dotenv()
    project_path = pathlib.Path(os.getenv("PROJECT_PATH"))
    path = project_path / "models" / "ann.pt"

    neurons = [6, 3]
    model = torch.nn.Sequential(
        torch.nn.Linear(neurons[0], neurons[1]),
        torch.nn.ReLU(),
        torch.nn.Linear(neurons[1], 1),
        torch.nn.Sigmoid(),
    )
    loss_fn = torch.nn.BCELoss()

    best_acc, acc_test, best_auc, auc_test = train(
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        model=model,
        path=path,
        lr=0.01,
        epoch_nr=200,
        loss_fn=loss_fn,
    )

    print(f"Train accuracy: {best_acc:.4f}")
    print(f"Train AUC: {best_auc:.4f}")
    print("-" * 50)
    print(f"Validation accuracy: {acc_test:.4f}")
    print(f"Validation AUC: {auc_test:.4f}")

    # Load the saved model's state_dict
    saved_model = torch.load(path)

    # Create a new instance of the model architecture
    loaded_model = torch.nn.Sequential(
        torch.nn.Linear(neurons[0], neurons[1]),
        torch.nn.ReLU(),
        torch.nn.Linear(neurons[1], 1),
        torch.nn.Sigmoid(),
    )

    # Load the saved state_dict into the new model instance
    loaded_model.load_state_dict(saved_model["model_state_dict"])

    # Evaluate on test set
    loss_test, acc_test, auc_test = evaluate_model(loaded_model, test_dataloader, loss_fn)
    print("-" * 50)
    print(f"Accuracy on test set: {acc_test:.4f}")
    print(f"AUC on test set: {auc_test:.4f}")


if __name__ == "__main__":
    main()
