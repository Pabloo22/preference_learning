"""Helper functions for preference learning.

This code has beem provided by the course instructors. Only minor naming
changes have been made to make it consistent with the rest of the code. MIT License does not
apply to this file.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from functools import partial
import pathlib
from typing import Union


class NumpyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.Tensor(data)
        self.targets = torch.LongTensor(targets.astype(int))

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


def regret(x, target):
    return torch.mean(
        torch.relu(-(target >= 1).float() * x) + torch.relu((target < 1).float() * x)
    )


def accuracy(x, target):
    return (target == (x[:, 0] > 0) * 1).detach().numpy().mean()


def auc_score(x, target):
    return roc_auc_score(target.detach().numpy(), x.detach().numpy()[:, 0])


def create_data_loader(x, y):
    dataset = NumpyDataset(x, y)
    return DataLoader(dataset, batch_size=len(dataset))


def train(model: torch.nn.Module,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          path: Union[str, pathlib.Path],
          lr: float = 0.01,
          epoch_nr: int = 200):
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
    best_acc = 0.0
    best_auc = 0.0
    for epoch in tqdm(range(epoch_nr)):
        for _, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = regret(outputs, labels)
            loss.backward()
            optimizer.step()
            acc = accuracy(outputs, labels)
            auc = auc(outputs, labels)

        if acc > best_acc:
            best_acc = acc
            best_auc = auc
            with torch.no_grad():
                for i, data in enumerate(test_dataloader, 0):
                    inputs, labels = data
                    outputs = model(inputs)
                    loss_test = regret(outputs, labels)
                    acc_test = accuracy(outputs, labels)
                    auc_test = auc(outputs, labels)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_train": loss,
                    "loss_test": loss_test,
                    "accuracy_train": acc,
                    "accuracy_test": acc_test,
                    "auc_train": auc,
                    "auc_test": auc_test,
                },
                path,
            )

    return best_acc, acc_test, best_auc, auc_test


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
