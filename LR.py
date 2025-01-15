import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import time


class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000, batch_size=32):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Ensure inputs are numpy arrays
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for iter in range(self.n_iters):
            # Shuffle data at the start of each iteration
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch gradient descent
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Linear combination and sigmoid function
                linear_model = np.dot(X_batch, self.weights) + self.bias
                y_predicted = self._sigmoid(linear_model)

                # Compute gradients
                dw = (1 / len(y_batch)) * np.dot(X_batch.T, (y_predicted - y_batch))
                db = (1 / len(y_batch)) * np.sum(y_predicted - y_batch)

                # Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            # Print progress and loss every 100 iterations
            if iter % 100 == 0:
                linear_model = np.dot(X, self.weights) + self.bias
                y_predicted = self._sigmoid(linear_model)
                loss = self._binary_cross_entropy(y, y_predicted)
                print(f"Iteration {iter}: Loss = {loss:.4f}")

    def predict(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return (y_predicted > 0.5).astype(int)

    def predict_proba(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def _sigmoid(self, x):
        # Clip values to avoid overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _binary_cross_entropy(self, y_true, y_pred):
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train_and_evaluate_lr():
    # Load the preprocessed BERT embeddings
    print("Loading data...")
    X_train = np.load("X_train_tfidf.npy")
    X_val = np.load("X_val_tfidf.npy")
    X_test = np.load("X_test_tfidf_new.npy")
    y_train = np.load("y_train_tfidf.npy")
    y_val = np.load("y_val_tfidf.npy")
    y_test = np.load("y_test_new.npy")

    # Initialize and train the model
    print("Training Logistic Regression...")
    start_time = time.time()

    model = LogisticRegression(
        learning_rate=0.01,
        n_iters=1000,
        batch_size=32
    )

    # Train the model
    model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Make predictions
    print("\nMaking predictions...")
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    # Calculate and print metrics
    print("\n=== Training Set Metrics ===")
    train_report = classification_report(y_train, train_preds, target_names=['Negative', 'Positive'])
    print("Classification Report:")
    print(train_report)

    print("\n=== Validation Set Metrics ===")
    val_report = classification_report(y_val, val_preds, target_names=['Negative', 'Positive'])
    print("Classification Report:")
    print(val_report)

    print("\n=== Test Set Metrics ===")
    test_report = classification_report(y_test, test_preds, target_names=['Negative', 'Positive'])
    print("Classification Report:")
    print(test_report)

    # Generate and plot confusion matrix for test set
    test_cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('lr_confusion_matrix.png')
    plt.close()

    # Save results
    results = {
        'training_time': training_time,
        'train_report': train_report,
        'validation_report': val_report,
        'test_report': test_report,
        'test_confusion_matrix': test_cm.tolist(),
        'hyperparameters': {
            'learning_rate': model.lr,
            'n_iterations': model.n_iters,
            'batch_size': model.batch_size
        }
    }

    with open('lr_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nResults have been saved to 'lr_results.json'")
    print("Confusion matrix plot has been saved as 'lr_confusion_matrix.png'")

    return model, results


if __name__ == "__main__":
    model, results = train_and_evaluate_lr()
