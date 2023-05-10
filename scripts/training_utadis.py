from sklearn.model_selection import train_test_split
import dotenv
import pathlib
import os
import torch
import functools

from preference_learning import (load_dataframe,
                                 create_data_loader,
                                 Uta,
                                 NormLayer,
                                 train,
                                 evaluate_model,
                                 set_seed,
                                 accuracy)

dotenv.load_dotenv()
MODEL_FILENAME = "ann_utadis.pt"
MODEL_PATH = pathlib.Path(os.getenv("PROJECT_PATH")) / "models" / MODEL_FILENAME

CRITERIA_NR = 6

# Hyperparameters
HIDDEN_NR = 30
EPOCHS = 200
LEARNING_RATE = 0.01


def main():
    set_seed(123)

    # Load data
    X_train, X_test, y_train, y_test = load_dataframe(mode="split")

    # Create Validation Set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create DataLoaders
    train_dataloader = create_data_loader(
        X_train.to_numpy().reshape(-1, 1, CRITERIA_NR),
        y_train.to_numpy().reshape(-1,))
    val_dataloader = create_data_loader(
        X_val.to_numpy().reshape(-1, 1, CRITERIA_NR),
        y_val.to_numpy().reshape(-1,))
    test_dataloader = create_data_loader(
        X_test.to_numpy().reshape(-1, 1, CRITERIA_NR),
        y_test.to_numpy().reshape(-1,))

    # Create Model
    uta = Uta(criteria_nr=CRITERIA_NR, hidden_nr=HIDDEN_NR)
    model = NormLayer(uta, CRITERIA_NR)

    acc_fn = functools.partial(accuracy, threshold=0)
    # Train Model
    best_acc, acc_test, best_auc, auc_test = train(
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        model=model,
        path=MODEL_PATH,
        lr=LEARNING_RATE,
        epoch_nr=EPOCHS,
        accuracy_fn=acc_fn,
    )
    # Print results
    print(f"Train accuracy: {best_acc:.4f}")
    print(f"Train AUC: {best_auc:.4f}")
    print("-" * 50)
    print(f"Validation accuracy: {acc_test:.4f}")
    print(f"Validation AUC: {auc_test:.4f}")
    # Load the saved model's state_dict
    saved_model = torch.load(MODEL_PATH)

    # Create a new instance of the model architecture
    loaded_model = NormLayer(Uta(criteria_nr=CRITERIA_NR, hidden_nr=HIDDEN_NR), CRITERIA_NR)

    # Load the state_dict into the new model
    loaded_model.load_state_dict(saved_model["model_state_dict"])

    # Evaluate on test set
    loss_test, acc_test, auc_test = evaluate_model(loaded_model, test_dataloader)
    print("-" * 50)
    print(f"Accuracy on test set: {acc_test:.4f}")
    print(f"AUC on test set: {auc_test:.4f}")


if __name__ == "__main__":
    main()
