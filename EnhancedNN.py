import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from testeval import *

X_train_tfidf = np.load("X_train_bert.npy")
X_test_tfidf = np.load("X_val_bert.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_val.npy")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)


# Datasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Increased batch size
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Simple Neural Network with Dropout

class EnhancedNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EnhancedNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True,
                            dropout=0.5)  # Dropout in lstm layers

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 256)  # Hidden size * 2 for bidirectional
        self.ln1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 512)
        self.ln2 = nn.LayerNorm(512)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(256, output_size)  # Output layer

    def forward(self, x):
        # LSTM expects input in the shape (batch_size, seq_len, input_size)
        # For non-sequential inputs, add a dummy sequence length dimension
        if x.ndim == 2:
            x = x.unsqueeze(1)  # Add sequence dimension: (batch_size, 1, input_size)

        # LSTM layer
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_size * 2)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step

        # Fully connected layers
        x = F.leaky_relu(self.fc1(lstm_out * 0.1))  # LeakyReLU with negative slope of 0.1
        x = self.ln1(x)
        x = self.dropout1(x)

        x = F.leaky_relu(self.fc2(x))
        x = self.ln2(x)
        x = self.dropout2(x)

        x = F.leaky_relu(self.fc3(x))
        x = self.ln3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x


# Initialize the model
hidden_size = 256  # Define the hidden layer size for the LSTM
num_layers = 1
input_size = X_train_tensor.shape[1]
output_size = len(torch.unique(y_train_tensor))
model = EnhancedNN(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers).to(
    device)

# Loss and optimizer with L2 regularization
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), 0.0001)  # Adam with L2 regularization
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # StepLR for decay
best_val_loss = float('inf')
patience = 5  # Number of epochs to wait for improvement
counter = 0
best_accuracy = 0.0
best_model_path = "models/EnhancedNN.pth"

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
num_epochs = 50
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)
grad_clip = 1
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    scheduler.step(val_loss)

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
    load_and_evaluate_model(
        model_class=EnhancedNN(),
        model_path="models/simpleNN.pth",
        X_test_path="X_test_bert.npy",
        y_test_path="y_test.npy",
    )
