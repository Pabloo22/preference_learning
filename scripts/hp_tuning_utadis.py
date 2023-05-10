import optuna
from sklearn.model_selection import train_test_split
import dotenv
import pathlib
import os
import functools

from preference_learning import (load_dataframe,
                                 create_data_loader,
                                 Uta,
                                 NormLayer,
                                 train,
                                 set_seed,
                                 accuracy)

dotenv.load_dotenv()
MODEL_FILENAME = "ann_utadis.pt"
MODEL_PATH = pathlib.Path(os.getenv("PROJECT_PATH")) / "models" / MODEL_FILENAME

CRITERIA_NR = 6

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


def main(trial):
    set_seed(123)

    # Hyperparameters
    HIDDEN_NR = trial.suggest_int("hidden_nr", 10, 100)
    EPOCHS = trial.suggest_int("epochs", 50, 500)
    LEARNING_RATE = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)

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
        save_model=False,
    )

    # Return the metric to be optimized (e.g., validation accuracy or loss)
    return acc_test


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(main, n_trials=50)

    print("Best trial:")
    trial_ = study.best_trial

    print(f"  Value: {trial_.value}")
    print("  Params: ")
    for key, value in trial_.params.items():
        print(f"    {key}: {value}")