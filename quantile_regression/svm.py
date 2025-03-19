from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch 
from torch.utils.data import DataLoader
import os, sys
sys.path.append(os.getcwd())
from quantile_regression.classification_model import PriceClassifier
from sklearn.metrics import classification_report
import numpy as np

class SVMClassifier(PriceClassifier):
    def fit(self, train_dataset, val_dataset, test_dataset):
        """
        Train the classification models using SVM.
        """
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        # Extract feature names from dataset
        self.feature_names = train_dataset.dataset.feature_names

        # Convert PyTorch dataset to numpy for training
        for x_short, x_long, y, mask, percent, label, _ in train_loader:
            x_long_train = x_long.numpy()
            label_train = label.numpy()

        for x_short, x_long, y, mask, percent, label, _ in val_loader:
            x_long_val = x_long.numpy()
            label_val = label.numpy()

        for x_short, x_long, y, mask, percent, label, _ in test_loader:
            x_long_test = x_long.numpy()
            label_test = label.numpy()

        print(f"Training size: {x_long_train.shape}, Validation size: {x_long_val.shape}, Test size: {x_long_test.shape}")

        num_classes = 3  # -10 to 10 -> 21 classes

        # Adjust labels from -10 to 10 to indices 0 to 20
        label_train_adj = label_train + num_classes // 2
        label_val_adj = label_val + num_classes // 2
        label_test_adj = label_test + num_classes // 2

        self.models = []  # Store models for each output

        for i in range(self.output_size):  # Create one model for each output
            print(f"Training SVM model for output {i}...")

            # Create an SVM pipeline with standard scaling
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced'))
            ])

            # Train the SVM model
            model.fit(x_long_train, label_train_adj[:, i])

            # Save the trained model
            self.models.append(model)

        self.save_model("best_svm_model.pkl")
        return self

    def evaluate(self, dataset, test_indices):
        """
        Evaluate the SVM models on the test dataset.
        """
        X = dataset.X_long_term[test_indices]
        label = dataset.labels[test_indices]

        for i, model in enumerate(self.models):
            print(f"Evaluating SVM model for output {i}...")
            preds = model.predict(X)
            true_labels = label[:, i]

            print("Classification Report:")
            print(classification_report(true_labels, preds))

    def forward(self, x_short, x_long):
        """
        Perform inference with SVM models.
        """
        x_long = x_long.cpu().numpy()

        predictions = []
        for model in self.models:
            preds = model.predict(x_long)
            predictions.append(preds)

        # Convert predictions to a PyTorch tensor
        predictions = np.column_stack(predictions)
        return torch.tensor(predictions, dtype=torch.float32)