import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def load_and_evaluate_model(
        model_class,
        model_path: str,
        X_test_path: str,
        y_test_path: str,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        display_confusion_matrix: bool = True
):
    """
    Load the model, test data, and evaluate the model on the test set. Prints evaluation metrics and misclassifications.

    Parameters:
        model_class (callable): The class of the model to instantiate.
        model_path (str): Path to the saved model file (state dictionary).
        X_test_path (str): Path to the numpy file containing test features.
        y_test_path (str): Path to the numpy file containing test labels.
        batch_size (int): Batch size for evaluation. Default is 32.
        device (str): Device to use ('cuda' or 'cpu'). Default is 'cuda' if available.
        display_confusion_matrix (bool): Whether to display the confusion matrix. Default is True.

    Returns:
        None: Prints test loss, accuracy, classification report, and optionally displays a confusion matrix.
    """
    # Load test data
    print("Loading test data...")
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    # Convert to PyTorch tensors and move to the device
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # Create DataLoader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the model architecture and state dictionary
    print("Loading model...")
    input_size = X_test_tensor.shape[1]
    output_size = len(torch.unique(y_test_tensor))
    model = model_class(input_size, output_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate the model
    print("Evaluating model...")
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    misclassified_samples = []
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            test_loss += criterion(outputs, y_batch).item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == y_batch)

            # Collect misclassified samples
            for i in range(len(preds)):
                if preds[i] != y_batch[i]:
                    misclassified_samples.append((X_batch[i].cpu().numpy(), preds[i].item(), y_batch[i].item()))

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = correct / len(test_dataset)

    # Print results
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(output_size)]))

    # Display confusion matrix
    if display_confusion_matrix:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(output_size)])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()



