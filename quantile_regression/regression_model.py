
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from sklearn.linear_model import LinearRegression
class PricePrediction(nn.Module):
    def save_model(self, model_path):
        """
        Lưu toàn bộ mô hình XGBoost vào file .pkl
        """
        joblib.dump(self, model_path)

    @staticmethod
    def load_model(model_path):
        """
        Load mô hình từ file .pkl
        """
        return joblib.load(model_path)


class PytorchQuantileRegression(PricePrediction):
    def fit(self, train_dataset, val_dataset, test_dataset):
        train_loader = DataLoader(train_dataset, batch_size=1024*32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # for batch_idx, (x_short, x_long, y, mask) in enumerate(train_loader):
        #     print(f"\nBatch {batch_idx + 1}:")
        #
        #     # Tính mean cho từng cột (feature) của x_short
        #     x_short_mean = x_short.mean(dim=(0, 1))  # Tính trung bình theo batch_size và window_size
        #     print("Mean of x_short features:")
        #     for i, val in enumerate(x_short_mean):
        #         print(f"  Feature {i}: {val.item():.4f}")
        #
        #     # Tính mean cho từng cột (feature) của x_long
        #     x_long_mean = x_long.mean(dim=0)  # Tính trung bình theo batch_size
        #     print("Mean of x_long features:")
        #     for i, val in enumerate(x_long_mean):
        #         print(f"  Feature {i}: {val.item():.4f}")
        #
        #     # Tính mean cho từng target (theo chiều thời gian)
        #     target_mean = y.mean(dim=0)  # Tính trung bình theo batch_size
        #     print("Mean of target values:")
        #     for i, val in enumerate(target_mean):
        #         print(f"  Target {i}: {val.item():.4f}")
        #     if random.random() < 0.3:
        #         break
        # exit()

        ##############################################
        # 3. Training Setup
        ##############################################
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        early_stopping = EarlyStopping(patience=8, delta=1e-4, path='best_model.pkl')

        num_epochs = 100

        train_losses, val_losses, test_losses = [], [], []

        ##############################################
        # 4. Training Loop
        ##############################################
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0

            for x_short, x_long, y, mask in train_loader:
                x_short, x_long, y, mask = x_short.to(device), x_long.to(device), y.to(device), mask.to(device)

                optimizer.zero_grad()

                pred_5, pred_50, pred_95 = model(x_short, x_long)

                loss_5 = quantile_loss(0.05, y, pred_5, mask)
                loss_50 = quantile_loss(0.50, y, pred_50, mask)
                loss_95 = quantile_loss(0.95, y, pred_95, mask)

                loss_total = loss_5 + loss_50 + loss_95

                loss_total.backward()
                optimizer.step()
                epoch_train_loss += loss_total.item()

            avg_train_loss = epoch_train_loss / len(train_loader)

            model.eval()
            epoch_val_loss = 0
            epoch_test_loss = 0  # Thêm biến tính loss cho tập test

            # Validation Loss
            with torch.no_grad():
                for x_short, x_long, y, mask in val_loader:
                    x_short, x_long, y, mask = x_short.to(device), x_long.to(device), y.to(device), mask.to(device)

                    pred_5, pred_50, pred_95 = model(x_short, x_long)

                    epoch_val_loss += quantile_loss(0.05, y, pred_5, mask).item()
                    epoch_val_loss += quantile_loss(0.5, y, pred_50, mask).item()
                    epoch_val_loss += quantile_loss(0.95, y, pred_95, mask).item()

            avg_val_loss = epoch_val_loss / len(val_loader)

            # Test Loss
            with torch.no_grad():
                for x_short, x_long, y, mask in test_loader:
                    x_short, x_long, y, mask = x_short.to(device), x_long.to(device), y.to(device), mask.to(device)

                    pred_5, pred_50, pred_95 = model(x_short, x_long)

                    epoch_test_loss += quantile_loss(0.05, y, pred_5, mask).item()
                    epoch_test_loss += quantile_loss(0.5, y, pred_50, mask).item()
                    epoch_test_loss += quantile_loss(0.95, y, pred_95, mask).item()

            avg_test_loss = epoch_test_loss / len(test_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            test_losses.append(avg_test_loss)  # Lưu lại loss tập test

            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"Stopped early at epoch {epoch+1}")
                break

        ##############################################
        # 5. Plot Learning Curve
        ##############################################
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.plot(test_losses, label='Test Loss')  # Thêm đường biểu diễn loss tập test
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.title('Learning Curve - 50% Quantile')
        plt.show()


# -------------------------------
# 1️⃣ Kernel-based Adaptive Network (KAN) Layer
# -------------------------------
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_kernels=16):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_kernels = num_kernels

        # Learnable parameters for kernel centers and widths
        self.centers = nn.Parameter(torch.randn(out_features, num_kernels))
        self.widths = nn.Parameter(torch.rand(out_features, num_kernels))

        # Linear projection for combining kernels
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        linear_out = self.linear(x)
        kernel_outputs = []
        for i in range(self.num_kernels):
            kernel = torch.exp(-((linear_out - self.centers[:, i]) ** 2) / (2 * self.widths[:, i] ** 2 + 1e-6))
            kernel_outputs.append(kernel)
        kernel_out = torch.stack(kernel_outputs, dim=-1).sum(dim=-1)
        return F.relu(kernel_out + linear_out)


# -------------------------------
# 2️⃣ KAN Block
# -------------------------------
class KANBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size=256, num_layers=4):
        super(KANBlock, self).__init__()
        self.kan_layers = nn.ModuleList([
            KANLayer(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.backcast_layer = nn.Linear(hidden_size, input_size)
        self.forecast_layer = nn.Linear(hidden_size, theta_size)

    def forward(self, x):
        for layer in self.kan_layers:
            x = layer(x)
        backcast = self.backcast_layer(x)
        forecast = self.forecast_layer(x)
        return backcast, forecast


# -------------------------------
# 3️⃣ Quantile Regressor with KAN
# -------------------------------
class KANQuantileRegressor(nn.Module):
    def __init__(self, output_size, num_blocks=3, hidden_size=256):
        super(KANQuantileRegressor, self).__init__()
        input_size = 153
        self.blocks = nn.ModuleList([
            KANBlock(input_size, output_size, hidden_size) for _ in range(num_blocks)
        ])
        self.output_5 = nn.Linear(output_size, output_size)
        self.output_50 = nn.Linear(output_size, output_size)
        self.output_95 = nn.Linear(output_size, output_size)

    def forward(self, x_short, x_long):
        x_short_flat = x_short.view(x_short.size(0), -1)
        combined_input = torch.cat((x_short_flat, x_long), dim=1)
        backcast = combined_input
        forecast = 0

        for block in self.blocks:
            block_backcast, block_forecast = block(backcast)
            backcast = backcast - block_backcast
            forecast += block_forecast

        pred_5 = self.output_5(forecast)
        pred_50 = self.output_50(forecast)
        pred_95 = self.output_95(forecast)
        return pred_5, pred_50, pred_95


# -------------------------------
# 4️⃣ LSTM-Based Quantile Regressor
# -------------------------------
class LSTMQuantileRegressor(PytorchQuantileRegression):
    def __init__(self, output_size, hidden_size=128, num_layers=2):
        super(LSTMQuantileRegressor, self).__init__()

        # ✅ Cập nhật input_size cho LSTM (short-term features): 5 cho features cơ bản, 166 cho TA features
        self.lstm = nn.LSTM(input_size=5 + 166, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # ✅ Cập nhật input_size = 3 cho feature cơ bản, 33 cho time features, 166 cho TA features, 16 cho timestep features
        self.fc_long = nn.Linear(3 + 33 + 166 + 16, hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.output_5 = nn.Linear(64, output_size)
        self.output_50 = nn.Linear(64, output_size)
        self.output_95 = nn.Linear(64, output_size)

    def forward(self, x_short, x_long):
        lstm_out, _ = self.lstm(x_short)
        lstm_out = lstm_out[:, -1, :]

        long_out = self.fc_long(x_long)
        combined = torch.cat((lstm_out, long_out), dim=1)
        out = self.fc(combined)

        pred_5 = self.output_5(out)
        pred_50 = self.output_50(out)
        pred_95 = self.output_95(out)
        return pred_5, pred_50, pred_95


# -------------------------------
# 5️⃣ Dense (Feedforward) Quantile Regressor
# -------------------------------
class DenseQuantileRegressor(nn.Module):
    def __init__(self, output_size):
        super(DenseQuantileRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(153, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.output_5 = nn.Linear(64, output_size)
        self.output_50 = nn.Linear(64, output_size)
        self.output_95 = nn.Linear(64, output_size)

    def forward(self, x_short, x_long):
        x_short_flat = x_short.view(x_short.size(0), -1)
        combined_input = torch.cat((x_short_flat, x_long), dim=1)

        out = self.fc(combined_input)
        pred_5 = self.output_5(out)
        pred_50 = self.output_50(out)
        pred_95 = self.output_95(out)
        return pred_5, pred_50, pred_95


class XGBoostQuantileRegressor(PricePrediction):
    def __init__(self, output_size):
        """
        XGBoost model for quantile regression using boosted quantile loss.
        """
        super().__init__()
        self.output_size = output_size  # Number of future steps (TARGET_LENGTH)
        self.models = {
            # 0.05: None,  # Models will be initialized during training
            0.50: None,
            # 0.95: None,
        }

        self.alpha = 1
        self.beta = 1

    def fit(self, train_dataset, val_dataset, test_dataset):
        """
        Train the XGBoost quantile regressors using boosted quantile loss.
        """
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        # Convert PyTorch dataset to numpy for XGBoost training
        for x_short, x_long, y, mask in train_loader:
            x_long = x_long.numpy()
            y = y.numpy()
            mask = mask.numpy()
            valid_indices = mask.sum(axis=1) > 0  # Select valid samples
            x_long_train = x_long[valid_indices]
            y_train = y[valid_indices]

        for x_short, x_long, y, mask in val_loader:
            x_long = x_long.numpy()
            y = y.numpy()
            mask = mask.numpy()
            valid_indices = mask.sum(axis=1) > 0
            x_long_val = x_long[valid_indices]
            y_val = y[valid_indices]

        for x_short, x_long, y, mask in test_loader:
            x_long = x_long.numpy()
            y = y.numpy()
            mask = mask.numpy()
            valid_indices = mask.sum(axis=1) > 0
            x_long_test = x_long[valid_indices]
            y_test = y[valid_indices]

        print(f"Training size: {x_long_train.shape}, {y_train.shape}, Validation size: {x_long_val.shape}, {y_val.shape}, Test size: {x_long_test.shape}, {y_test.shape}")

        # ✅ Train separate models for each quantile using `boosted_quantile_loss`
        for q in self.models:

            print(f"Training quantile {q}: {y_train.shape}, Validation quantile {q}: {y_val.shape}")

            def custom_weight(y):
                # return 1 + beta * (np.abs(y[:, 0]) ** alpha)  # Increase impact of large y
                return 1 + beta * (np.max(np.abs(y), axis=1) ** alpha)  # Increase impact of large y

            alpha = 2.5  # Exponential factor for boosting
            beta = 2.0   # Scaling factor

            dtrain = xgb.DMatrix(x_long_train, label=y_train, weight=custom_weight(y_train))  # Use absolute y as weights
            dval = xgb.DMatrix(x_long_val, label=y_val, weight=custom_weight(y_val))
            dtest = xgb.DMatrix(x_long_test, label=y_test, weight=custom_weight(y_test))


            self.models[q] = xgb.train(
                params={
                    "objective": "reg:squarederror",  # ✅ Use squared error for RMSE
                    "learning_rate": 0.1,             # ✅ Standard learning rate
                    "max_depth": 6,                   # ✅ Tree depth
                    "n_estimators": 300,             # ✅ Boosting rounds
                    "eval_metric": "rmse",            # ✅ Use RMSE as evaluation metric
                },
                dtrain=dtrain,
                num_boost_round=1000,
                evals=[(dtest, "test"), (dtrain, "train"), (dval, "validation")],
                early_stopping_rounds=15,  # ✅ Stop early if validation loss stops improving
                verbose_eval=True
            )


        self.save_model("best_model.pkl")

    def custom_loss(self, y_pred, dtrain, q):
        """
        Custom training loss for XGBoost using boosted quantile loss.
        """
        y_true = dtrain.get_label().reshape(y_pred.shape)
        grad, hess = boosted_quantile_loss(q, y_true, y_pred, alpha=self.alpha, beta=self.beta)
        return grad, hess

    def custom_eval(self, y_pred, dtrain, q):
        """
        Custom evaluation function for XGBoost.
        Uses `quantile_loss` with outlier emphasis.
        """
        y_true = dtrain.get_label().reshape(y_pred.shape)

        # ✅ Convert numpy arrays to PyTorch tensors
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        # ✅ Create a mask of all 1s (XGBoost does not provide padding info)
        mask = (y_true != 0).astype(np.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        # ✅ Compute loss using the custom function
        loss_value = quantile_loss(q, y_true_tensor, y_pred_tensor, mask_tensor, alpha=self.alpha, beta=self.beta).item()

        return "quantile_loss", loss_value  # ✅ Lower is better

    def forward(self, x_short, x_long):
        """
        Perform inference with XGBoost quantile models.
        Mimics the PyTorch model's `forward` method.
        """
        x_long = x_long.cpu().numpy()  # Convert tensor to numpy

        # ✅ Convert to DMatrix before predicting
        dtest = xgb.DMatrix(x_long)

        # ✅ Predict `output_size` values per sample for each quantile
        pred_5 = self.models[0.5].predict(dtest)  # Shape: (batch_size, output_size)
        pred_50 = self.models[0.50].predict(dtest)
        pred_95 = self.models[0.5].predict(dtest)

        # Convert to PyTorch tensors and return
        return (
            torch.tensor(pred_5, dtype=torch.float32),  # Shape: (batch_size, output_size)
            torch.tensor(pred_50, dtype=torch.float32),
            torch.tensor(pred_95, dtype=torch.float32),
        )

class LinearQuantileRegressor(PricePrediction):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.models = {
            0.05: LinearRegression(),
            0.50: LinearRegression(),
            0.95: LinearRegression()
        }

    def fit(self, train_dataset, val_dataset, test_dataset):
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        
        # Convert PyTorch dataset to numpy for sklearn training
        for x_short, x_long, y, mask in train_loader:
            x_long = x_long.numpy()
            y = y.numpy()
            mask = mask.numpy()
            valid_indices = mask.sum(axis=1) > 0  # Select valid samples
            x_long_train = x_long[valid_indices]
            y_train = y[valid_indices]

        # Train separate models for each quantile
        for q, model in self.models.items():
            if q == 0.50:
                model.fit(x_long_train, y_train)
            else:
                # For 0.05 and 0.95 quantiles, we'll adjust the predictions later
                model.fit(x_long_train, y_train)

        self.save_model("best_linear_model.pkl")

    def forward(self, x_short, x_long):
        x_long = x_long.cpu().numpy()

        pred_5 = self.models[0.05].predict(x_long)
        pred_50 = self.models[0.50].predict(x_long)
        pred_95 = self.models[0.95].predict(x_long)

        # Adjust predictions for 0.05 and 0.95 quantiles
        pred_5 = pred_50 - 1.96 * np.std(pred_50, axis=0)
        pred_95 = pred_50 + 1.96 * np.std(pred_50, axis=0)

        return (
            torch.tensor(pred_5, dtype=torch.float32),
            torch.tensor(pred_50, dtype=torch.float32),
            torch.tensor(pred_95, dtype=torch.float32),
        )

    def save_model(self, model_path):
        joblib.dump(self.models, model_path)

    @staticmethod
    def load_model(model_path):
        return joblib.load(model_path)

# -------------------------------
# 6️⃣ Quantile Loss Function
# -------------------------------
def quantile_loss_me(q, y_true, y_pred, mask):
    e = y_true - y_pred
    loss = torch.max(q * e, (q - 1) * e)
    loss = loss * mask
    valid_count = mask.sum() + 1e-8
    return loss.sum() / valid_count

def quantile_loss(q, y_true, y_pred, mask, alpha=2.5, beta=2):
    """
    Boosted Quantile Loss to emphasize outliers.

    Args:
        q (float): Quantile level (e.g., 0.05, 0.50, 0.95)
        y_true (tensor): Ground truth values
        y_pred (tensor): Predicted values
        mask (tensor): Mask for valid values (1 for valid, 0 for padding)
        alpha (float): Exponentiation factor for boosting large errors
        beta (float): Factor to control loss weighting on outliers

    Returns:
        Tensor: Scaled quantile loss with outlier emphasis
    """
    e = y_true - y_pred  # Compute error
    base_loss = torch.max(q * e, (q - 1) * e)  # Standard quantile loss

    # ✅ Apply exponential boosting for large errors
    # boosted_loss = base_loss * (1 + beta * torch.abs(e) ** alpha)
    boosted_loss = base_loss * (1 + beta * torch.abs(y_true) ** alpha)

    # ✅ Apply mask to exclude padded values
    boosted_loss = boosted_loss * mask

    # ✅ Normalize by the number of valid values
    valid_count = mask.sum() + 1e-8
    return boosted_loss.sum() / valid_count

def boosted_quantile_loss(q, y_true, y_pred, alpha=4.5, beta=4.5):
    """
    Custom quantile loss with outlier boosting for XGBoost.

    Args:
        q (float): Quantile level (e.g., 0.05, 0.50, 0.95).
        y_true (array): Ground truth values.
        y_pred (array): Predicted values.
        alpha (float): Exponentiation factor for boosting large errors.
        beta (float): Factor to control loss weighting on outliers.

    Returns:
        grad (array): First derivative (gradient).
        hess (array): Second derivative (hessian).
    """
    e = y_true - y_pred
    loss = np.maximum(q * e, (q - 1) * e)

    # Apply boosting for large errors
    weight = (1 + beta * np.abs(e) ** alpha)

    grad = np.where(e > 0, q, q - 1) * weight  # Gradient (first derivative)
    hess = np.ones_like(y_pred) * weight  # Use adaptive hessian like XGBoost
    
    return grad, hess

# -------------------------------
# 7️⃣ Early Stopping Mechanism
# -------------------------------
class EarlyStopping:
    def __init__(self, patience=7, delta=1e-4, path='best_model.pkl'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            model.save_model(self.path)  # ✅ Save the entire model (with architecture)
            print(f"Best model saved with val_loss = {val_loss:.4f}")
        else:
            self.counter += 1
            print(f"No improvement for {self.counter} epoch(s).")
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                self.early_stop = True




# -------------------------------
# 8 GRUQuantileRegressor
# -------------------------------

class GRUQuantileRegressor(PytorchQuantileRegression):
    def __init__(self, output_size, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=True):
        super(GRUQuantileRegressor, self).__init__()
        
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=5 + 166, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0, 
            bidirectional=bidirectional
        )
        
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.fc_long = nn.Sequential(
            nn.Linear(3 + 33 + 166 + 16, hidden_size),
            nn.BatchNorm1d(hidden_size),  
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size), 
            nn.ReLU(inplace=True)
        )

        combined_size = gru_output_size + hidden_size

        self.fc = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.BatchNorm1d(128),  
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.output_5 = nn.Linear(64, output_size)
        self.output_50 = nn.Linear(64, output_size)
        self.output_95 = nn.Linear(64, output_size)

    def forward(self, x_short, x_long):
        gru_out, _ = self.gru(x_short)
        gru_out_last = gru_out[:, -1, :]  

        long_out = self.fc_long(x_long)  

        combined = torch.cat((gru_out_last, long_out), dim=1)
        out = self.fc(combined)

        pred_5 = self.output_5(out)
        pred_50 = self.output_50(out)
        pred_95 = self.output_95(out)

        return pred_5, pred_50, pred_95


 
# -------------------------------
# 9 CNNQuantileRegressor
# -------------------------------

class CNNQuantileRegressor(PytorchQuantileRegression):
    def __init__(self, output_size, num_channels=64, kernel_size=3, dropout=0.3):
        super(CNNQuantileRegressor, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=5 + 166, out_channels=num_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels * 2, kernel_size=kernel_size, padding=1)
        self.conv3 = nn.Conv1d(in_channels=num_channels * 2, out_channels=num_channels * 4, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        
        self.fc_long = nn.Sequential(
            nn.Linear(3 + 33 + 166 + 16, num_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(num_channels * 4, num_channels * 4),
            nn.ReLU(inplace=True)
        )
        
     
        combined_size = num_channels * 4 + num_channels * 4
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
      
        self.output_5 = nn.Linear(64, output_size)
        self.output_50 = nn.Linear(64, output_size)
        self.output_95 = nn.Linear(64, output_size)
        
    def forward(self, x_short, x_long):
        
        x_short = x_short.permute(0, 2, 1)  # Change to (batch, channels, sequence_length) for CNN
        conv_out = self.relu(self.conv1(x_short))
        conv_out = self.relu(self.conv2(conv_out))
        conv_out = self.relu(self.conv3(conv_out))
        conv_out = torch.mean(conv_out, dim=2)  # Global Average Pooling
        
        # Long-term feature extraction
        long_out = self.fc_long(x_long)
        
        # Combine CNN and long-term features
        combined = torch.cat((conv_out, long_out), dim=1)
        out = self.fc(combined)
        
        # Predict quantiles
        pred_5 = self.output_5(out)
        pred_50 = self.output_50(out)
        pred_95 = self.output_95(out)
        
        return pred_5, pred_50, pred_95
    

# -------------------------------
# RNNQuantileRegressor
# -------------------------------

import torch
import torch.nn as nn

class RNNQuantileRegressor(PytorchQuantileRegression):
    def __init__(self, output_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(RNNQuantileRegressor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN Layer (Thay thế CNN)
        self.rnn = nn.RNN(input_size=5 + 166, 
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout,
                          nonlinearity='relu')  # Dùng ReLU thay vì tanh

        self.dropout = nn.Dropout(dropout)

        # Fully connected layers cho đầu vào dài hạn
        self.fc_long = nn.Sequential(
            nn.Linear(3 + 33 + 166 + 16, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )

        # Kết hợp RNN output và đầu vào dài hạn
        combined_size = hidden_size + hidden_size
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Các lớp đầu ra cho dự đoán 5%, 50%, 95%
        self.output_5 = nn.Linear(64, output_size)
        self.output_50 = nn.Linear(64, output_size)
        self.output_95 = nn.Linear(64, output_size)

    def forward(self, x_short, x_long):
        # RNN expects (batch, seq_length, input_size), input hiện có dạng (batch, seq_length, channels)
        # Giữ nguyên thứ tự đầu vào (batch, seq_length, input_size)
        rnn_out, _ = self.rnn(x_short)  # Output có shape (batch, seq_length, hidden_size)

        # Lấy thông tin từ bước cuối cùng của RNN (Last Time Step)
        rnn_out = rnn_out[:, -1, :]  # (batch, hidden_size)

        # Dữ liệu dài hạn
        long_out = self.fc_long(x_long)

        # Kết hợp RNN output và long-term input
        combined = torch.cat((rnn_out, long_out), dim=1)
        out = self.fc(combined)

        # Dự đoán phân vị
        pred_5 = self.output_5(out)
        pred_50 = self.output_50(out)
        pred_95 = self.output_95(out)

        return pred_5, pred_50, pred_95

     
# -------------------------------
# DeepFeedForwardQuantileRegressor
# -------------------------------
class DeepFeedforwardRegressor(PytorchQuantileRegression):
    def __init__(self, output_size, hidden_size=128, num_layers=4, dropout=0.3):
        super(DeepFeedforwardRegressor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Fully connected layers cho đầu vào ngắn hạn
        self.fc_short = nn.Sequential(
            nn.Linear(5 + 166, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            *[nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)) for _ in range(num_layers - 1)]
        )
        
        self.fc_long = nn.Sequential(
            nn.Linear(3 + 33 + 166 + 16, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            *[nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)) for _ in range(num_layers - 1)]
        )
        
        combined_size = hidden_size + hidden_size
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.output_5 = nn.Linear(64, output_size)
        self.output_50 = nn.Linear(64, output_size)
        self.output_95 = nn.Linear(64, output_size)
    
    def forward(self, x_short, x_long):
        short_out = self.fc_short(x_short)  
        if len(short_out.shape) == 3:
            short_out = short_out[:, -1, :]  
        
      
        long_out = self.fc_long(x_long)  
        combined = torch.cat((short_out, long_out), dim=1) 
        out = self.fc(combined)

        pred_5 = self.output_5(out)
        pred_50 = self.output_50(out)
        pred_95 = self.output_95(out)

        return pred_5, pred_50, pred_95

