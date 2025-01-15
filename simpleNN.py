import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import testeval as te
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load Data
X_train_tfidf = np.load("X_train_bert.npy")
X_test_tfidf = np.load("X_val_bert.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_val.npy")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#
#
# Initialize the model

input_size = X_train_tensor.shape[1]
output_size = len(torch.unique(y_train_tensor))
model = SimpleNN(input_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Early Stopping and Best Model Saving
best_val_loss = float('inf')
patience = 3
counter = 0
best_model_path = "models/simpleNN.pth"
best_accuracy = 0.0

# For plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop with Early Stopping
num_epochs = 50
for epoch in tqdm(range(num_epochs), desc="Training"):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == y_batch).item()
        total += y_batch.size(0)

    train_loss = epoch_loss / len(train_loader)
    train_accuracy = correct / total

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation after each epoch
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            val_loss += criterion(outputs, y_batch).item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == y_batch).item()
            total += y_batch.size(0)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_loss /= len(test_loader)
    val_accuracy = correct / total

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Early stopping
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        counter = 0
        torch.save(model.state_dict(), best_model_path)  # Save best model
        print(f"Epoch {epoch + 1}: accuracy improved to {val_accuracy:.4f}. Model saved.")
    else:
        counter += 1
        print(f"Epoch {epoch + 1}: accuracy did not improve. Counter: {counter}")

    if counter >= patience:
        print("Early stopping triggered.")
        break

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(output_size)])
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
    print(report)

# Plotting training and validation metrics
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim([min(min(train_losses), min(val_losses)) * 0.9, max(max(train_losses), max(val_losses)) * 1.1])
plt.title("Loss vs Epoch")
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([min(min(train_accuracies), min(val_accuracies)) * 0.9, 1.0])
plt.title("Accuracy vs Epoch")
plt.legend()

plt.tight_layout()
plt.show()

print("Training complete.")

if __name__ == "__main__":
    te.load_and_evaluate_model(
        model_class=SimpleNN,
        model_path="models/simpleNN.pth",
        X_test_path="X_test_bert.npy",
        y_test_path="y_test.npy",
    )
