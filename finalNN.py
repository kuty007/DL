import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns


class BERTSentimentClassifier(nn.Module):
    def __init__(self, bert_embedding_dim=768, dropout_rate=0.3):
        super(BERTSentimentClassifier, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(bert_embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.output = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=50, device='cuda', patience=7):
    # Initialize tracking variables
    best_accuracy = 0
    counter = 0
    best_model_path = 'best_model.pt'

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    model = model.to(device)
    start_time = time.time()

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Training phase
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = epoch_loss / len(train_loader)
        train_accuracy = correct / total

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1)

                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = correct / total

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Early stopping check
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': epoch,
            }, best_model_path)
            print(f"Epoch {epoch + 1}: accuracy improved to {val_accuracy:.4f}. Model saved.")
        else:
            counter += 1
            print(f"Epoch {epoch + 1}: accuracy did not improve. Counter: {counter}")

        if counter >= patience:
            print("Early stopping triggered.")
            break

        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print("Classification Report:")
        print(report)

    # Plot training history
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim([min(min(train_losses), min(val_losses)) * 0.9,
              max(max(train_losses), max(val_losses)) * 1.1])
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
    plt.savefig('training_history.png')
    plt.show()

    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies,
        'best_accuracy': best_accuracy,
        'epochs_trained': len(train_losses),
        'training_time': time.time() - start_time
    }

    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=4)

    print("Training complete.")
    return history


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on test data and generate detailed metrics.
    """
    model.eval()
    test_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)

            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            predicted = (outputs > 0.5).float()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average test loss
    test_loss /= len(test_loader)

    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Create detailed report
    report = classification_report(all_labels, all_predictions, target_names=['Negative', 'Positive'])

    return {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }


def plot_confusion_matrix(cm):
    """
    Plot confusion matrix using seaborn.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load all data
    X_train = torch.FloatTensor(np.load("X_train_bert.npy"))
    X_val = torch.FloatTensor(np.load("X_val_bert.npy"))
    X_test = torch.FloatTensor(np.load("X_test_bert.npy"))
    y_train = torch.FloatTensor(np.load("y_train.npy"))
    y_val = torch.FloatTensor(np.load("y_val.npy"))
    y_test = torch.FloatTensor(np.load("y_test.npy"))

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model and training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERTSentimentClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # Train the model
    print("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=50,
        patience=7,
        device=device
    )

    # Load best model for testing
    print("\nLoading best model for testing...")
    best_model = BERTSentimentClassifier().to(device)
    checkpoint = torch.load('best_model.pt')
    best_model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_model(best_model, test_loader, criterion, device)

    # Print test results
    print("\n=== Test Results ===")
    print(f"Test Loss: {test_results['test_loss']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    print(f"F1 Score: {test_results['f1']:.4f}")

    print("\nClassification Report:")
    print(test_results['classification_report'])

    # Plot confusion matrix
    plot_confusion_matrix(test_results['confusion_matrix'])

    # Save test results
    results = {
        'test_metrics': {
            'loss': float(test_results['test_loss']),
            'accuracy': float(test_results['accuracy']),
            'precision': float(test_results['precision']),
            'recall': float(test_results['recall']),
            'f1': float(test_results['f1']),
        },
        'confusion_matrix': test_results['confusion_matrix'].tolist(),
        'classification_report': test_results['classification_report'],
        'training_history': history
    }

    with open('final_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nResults have been saved to 'final_results.json'")
    print("Confusion matrix plot has been saved as 'confusion_matrix.png'")


if __name__ == "__main__":
    main()