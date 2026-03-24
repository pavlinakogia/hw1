import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


# 1. Ορίζουμε την κλάση ΕΞΩ από τη συνάρτηση (Top Level)
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# 2. Ορίζουμε τον Wrapper ΕΞΩ από τη συνάρτηση για να είναι Picklable
class NNWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            probs = self.model(X_t).numpy()
        return (probs > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            probs = self.model(X_t).numpy()
        return np.hstack([1 - probs, probs])


def train_neural_network(X_train, y_train, X_val, y_val):
    # Μετατροπή σε Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    # Εξασφαλίζουμε ότι τα y είναι αριθμητικά (0/1)
    y_train_t = torch.tensor(y_train.values.astype(np.float32), dtype=torch.float32).view(-1, 1)
    y_val_t = torch.tensor(y_val.values.astype(np.float32), dtype=torch.float32).view(-1, 1)

    model = NeuralNet(X_train_t.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    best_loss = float("inf")
    patience = 7
    counter = 0

    if not os.path.exists("models"): os.makedirs("models")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "models/neural_network.pt")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Φόρτωση των καλύτερων βαρών
    model.load_state_dict(torch.load("models/neural_network.pt", weights_only=True))

    # Επιστρέφουμε τον Wrapper (τώρα είναι πλέον picklable!)
    return NNWrapper(model)