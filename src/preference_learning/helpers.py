"""Helper functions for preference learning.

This code has beem provided by the lab instructor Krzysztof Martyn. Only minor
changes have been made to make it consistent with the rest of the code. MIT License may not
apply to this file.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from functools import partial
import pathlib
from typing import Union, Tuple, Callable


class NumpyDataset(Dataset):
    """Dataset wrapping numpy arrays."""
    def __init__(self, data: np.ndarray, targets: np.ndarray):
        self.data = torch.Tensor(data)
        self.targets = torch.LongTensor(targets.astype(int))

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


def regret(x, target):
    """Returns the regret for a given model output and target.

    The regret is defined as the negative of the loss function. In this case,
    the loss function is the hinge loss.
    """
    return torch.mean(
        torch.relu(-(target >= 1).float() * x) + torch.relu((target < 1).float() * x)
    )


def accuracy(x, target, threshold=0.5):
    """Returns the accuracy for a given model output and target.

    The accuracy is defined as the percentage of correct predictions. In this
    case, the prediction is correct if the model output is greater than 0.
    """
    return (target == (x[:, 0] > threshold) * 1).detach().numpy().mean()


def auc_score(x, target):
    return roc_auc_score(target.detach().numpy(), x.detach().numpy()[:, 0])


def create_data_loader(x, y, batch_size=None):
    """Creates a data loader from numpy arrays.

    Slightly modified version of the original function.
    """
    if batch_size is None:
        batch_size = len(x)
    dataset = NumpyDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(model: torch.nn.Module,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          path: Union[str, pathlib.Path],
          lr: float = 0.01,
          epoch_nr: int = 200,
          loss_fn: Callable = regret,
          accuracy_fn: Callable = accuracy,
          save_model: bool = True
          ) -> Tuple[float, float, float, float]:
    """Trains a model and saves the best model to a file.

    The model is trained using the AdamW optimizer. The best model is saved to
    a file specified by the path argument. The model is saved if the test
    accuracy is greater than the previous best accuracy.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
    best_acc = 0.0
    best_auc = 0.0
    for epoch in tqdm(range(epoch_nr)):
        for _, data in enumerate(train_dataloader):
            inputs, labels = data
            labels = labels.float()  # Convert labels to float
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs_ = outputs[:, 0]
            loss = loss_fn(outputs_, labels)
            loss.backward()
            optimizer.step()

        loss_train, acc_train, auc_train = evaluate_model(model, train_dataloader, loss_fn, accuracy_fn)
        loss_test, acc_test, auc_test = evaluate_model(model, test_dataloader, loss_fn, accuracy_fn)

        if acc_test > best_acc:
            best_acc = acc_test
            best_auc = auc_test
            if save_model:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss_train": loss_train,
                        "loss_test": loss_test,
                        "accuracy_train": acc_train,
                        "accuracy_test": acc_test,
                        "auc_train": auc_train,
                        "auc_test": auc_test,
                    },
                    path,
                )
                print("Saved model to", path)
                print("Epoch: ", epoch)

    return best_acc, acc_test, best_auc, auc_test


def evaluate_model(model: torch.nn.Module,
                   dataloader: DataLoader,
                   loss_fn: Callable = regret,
                   accuracy_fn: Callable = accuracy) -> Tuple[float, float, float]:
    """Returns loss, accuracy and AUC for a given model and data loader."""
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_acc = 0.0
    total_auc = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            labels = labels.float()  # Convert labels to float
            outputs = model(inputs)
            outputs_ = outputs[:, 0]
            loss = loss_fn(outputs_, labels)
            acc = accuracy_fn(outputs, labels)
            auc = auc_score(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total_acc += acc * inputs.size(0)
            total_auc += auc * inputs.size(0)
            num_samples += inputs.size(0)

    return total_loss / num_samples, total_acc / num_samples, total_auc / num_samples


class Hook:
    def __init__(self, m, f):
        self.hook = m.register_forward_hook(partial(f, self))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


def append_output(hook, mod, inp, outp):
    if not hasattr(hook, "stats"):
        hook.stats = []
    if not hasattr(hook, "name"):
        hook.name = mod.__class__.__name__
    data = hook.stats
    data.append(outp.data)
