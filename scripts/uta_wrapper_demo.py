from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import dotenv
import pathlib
import os

from preference_learning import load_dataframe, UtaWrapper


def main():
    # Load data
    X_train, X_test, y_train, y_test = load_dataframe(mode="split")

    # Create Validation Set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize and train the UtaWrapper model
    uta_wrapper = UtaWrapper()
    uta_wrapper.fit(X_train, y_train)

    # Make predictions on the validation set
    val_preds = uta_wrapper.predict(X_val)
    val_probs = uta_wrapper.predict_proba(X_val)

    # Calculate accuracy and AUC
    val_accuracy = accuracy_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_probs)

    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    # Make predictions on the test set
    test_preds = uta_wrapper.predict(X_test)
    test_probs = uta_wrapper.predict_proba(X_test)

    # Calculate accuracy and AUC
    test_accuracy = accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)

    print("-" * 50)
    print(f"Accuracy on test set: {test_accuracy:.4f}")
    print(f"AUC on test set: {test_auc:.4f}")

    # Save the model
    dotenv.load_dotenv()
    path = pathlib.Path(os.getenv("PROJECT_PATH")) / "models" / "ann_utadis.pt"

    uta_wrapper.save_model(path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    main()
