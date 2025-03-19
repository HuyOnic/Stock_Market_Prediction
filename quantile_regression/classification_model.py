# ------------------------------
# 8️⃣ Classification Model
# -------------------------------

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
import lightgbm as lgb
import catboost as cb
import sys
import os
import optuna

from regression_model import PricePrediction
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import precision_score
import datetime
import json


# Get the absolute path of the dataset directory
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
sys.path.append(dataset_path)

from features_cal_tg_and_step import label_to_percent
from features_get_encoded_feature_name import get_origin_feature_name

class PriceClassifier(PricePrediction):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.models = []
        self.feature_names = None

    def prepare_data(self, train_dataset, val_dataset, test_dataset):
        """Chuyển dữ liệu từ PyTorch DataLoader sang NumPy"""
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        for _, x_long, y, mask, _, label, _ in train_loader:
            self.x_long_train, self.y_train, self.mask_train, self.label_train = x_long.numpy(), y.numpy(), mask.numpy(), label.numpy()

        for _, x_long, y, mask, _, label, _ in val_loader:
            self.x_long_val, self.y_val, self.mask_val, self.label_val = x_long.numpy(), y.numpy(), mask.numpy(), label.numpy()

        for _, x_long, y, mask, _, label, _ in test_loader:
            self.x_long_test, self.y_test, self.mask_test, self.label_test = x_long.numpy(), y.numpy(), mask.numpy(), label.numpy()

        self.feature_names = train_dataset.dataset.feature_names
        print(f"Training size: {self.x_long_train.shape}, Validation size: {self.x_long_val.shape}, Test size: {self.x_long_test.shape}")

        num_classes = 3  # -10 đến 10 -> 21 lớp

        # Chuyển label từ -10 đến 10 thành index từ 0 đến 20
        self.label_train = self.label_train + num_classes//2
        self.label_val = self.label_val + num_classes//2
        self.label_test = self.label_test + num_classes//2


class NNClassifier(PricePrediction):
    def __init__(self, output_size):
        """
        Classification model for quantile prediction using categorical outputs.
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
        Train the classification models using similar structure to XGBoost.
        """
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        # Convert PyTorch dataset to numpy for training
        for x_short, x_long, y, mask, percent, label in train_loader:
            x_long = x_long.numpy()
            label = label.numpy()
            mask = mask.numpy()
            valid_indices = mask.sum(axis=1) > 0  # Select valid samples
            x_long_train = x_long[valid_indices]
            label_train = label[valid_indices]

        for x_short, x_long, y, mask, percent, label in val_loader:
            x_long = x_long.numpy()
            label = label.numpy()
            mask = mask.numpy()
            valid_indices = mask.sum(axis=1) > 0
            x_long_val = x_long[valid_indices]
            label_val = label[valid_indices]

        for x_short, x_long, y, mask, percent, label in test_loader:
            x_long = x_long.numpy()
            label = label.numpy()
            mask = mask.numpy()
            valid_indices = mask.sum(axis=1) > 0
            x_long_test = x_long[valid_indices]
            label_test = label[valid_indices]

        print(f"Training size: {x_long_train.shape}, {label_train.shape}, Validation size: {x_long_val.shape}, {label_val.shape}, Test size: {x_long_test.shape}, {label_test.shape}")

        # Train separate models for each quantile
        for q in self.models:
            print(f"Training quantile {q}: {label_train.shape}, Validation quantile {q}: {label_val.shape}")

            def custom_weight(y):
                return 1 + self.beta * (np.max(np.abs(y), axis=1)) ** self.alpha

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize neural network for this quantile
            model = nn.Sequential(
                nn.Linear(x_long_train.shape[1], 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 21 * self.output_size)  # 21 classes (-10 to 10) for each output step
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Convert data to PyTorch tensors
            x_train_tensor = torch.FloatTensor(x_long_train).to(device)
            label_train_tensor = torch.LongTensor(label_train + 10).to(device)  # Shift labels to 0-20 range
            x_val_tensor = torch.FloatTensor(x_long_val).to(device)
            label_val_tensor = torch.LongTensor(label_val + 10).to(device)
            
            best_val_loss = float('inf')
            patience = 15
            patience_counter = 0
            
            for epoch in range(1000):
                # Training
                model.train()
                optimizer.zero_grad()
                outputs = model(x_train_tensor)
                outputs = outputs.view(-1, self.output_size, 21)
                
                train_loss = 0
                for t in range(self.output_size):
                    train_loss += criterion(outputs[:, t, :], label_train_tensor[:, t])
                
                train_loss.backward()
                optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(x_val_tensor)
                    val_outputs = val_outputs.view(-1, self.output_size, 21)
                    
                    val_loss = 0
                    for t in range(self.output_size):
                        val_loss += criterion(val_outputs[:, t, :], label_val_tensor[:, t])
                print("helo")
                print(f"Epoch {epoch+1}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.models[q] = model.state_dict()  # Save best model
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        self.save_model("best_classification_model.pkl")
        return self

    def forward(self, x_short, x_long):
        """
        Perform inference with classification models.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_long = x_long.to(device)
        
        predictions = {}
        for q in self.models:
            # Initialize model architecture
            model = nn.Sequential(
                nn.Linear(x_long.shape[1], 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 21 * self.output_size)
            ).to(device)
            
            # Load saved weights
            model.load_state_dict(self.models[q])
            model.eval()
            
            with torch.no_grad():
                output = model(x_long)
                output = output.view(-1, self.output_size, 21)
                output = F.softmax(output, dim=2)
                pred_classes = torch.argmax(output, dim=2).float() - 10
                predictions[q] = pred_classes

        return (
            predictions.get(0.05, predictions[0.50]),  # Use 0.50 as fallback
            predictions[0.50],
            predictions.get(0.95, predictions[0.50])   # Use 0.50 as fallback
        )
    
class XGBoostClassifier(PriceClassifier):

    def fit(self, train_dataset, val_dataset, test_dataset):
        """
        Train the classification models using similar structure to XGBoost.
        """
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        # Extract feature names from dataset
        self.feature_names = train_dataset.dataset.feature_names

        # Convert PyTorch dataset to numpy for training
        for x_short, x_long, y, mask, percent, label, _ in train_loader:
            x_long_train = x_long.numpy()
            y_train = y.numpy()
            mask = mask.numpy()
            label_train = label.numpy()

            # valid_indices = mask.sum(axis=1) > 0  # Select valid samples
            # x_long_train = x_long[valid_indices]
            # y_train = y[valid_indices]
            # label_train = label[valid_indices]

        for x_short, x_long, y, mask, percent, label, _ in val_loader:
            x_long_val = x_long.numpy()
            y_val = y.numpy()
            mask = mask.numpy()
            label_val = label.numpy()

            # valid_indices = mask.sum(axis=1) > 0
            # x_long_val = x_long[valid_indices]
            # y_val = y[valid_indices]
            # label_val = label[valid_indices]

        for x_short, x_long, y, mask, percent, label, _ in test_loader:
            x_long_test = x_long.numpy()
            y_test = y.numpy()
            mask = mask.numpy()
            label_test = label.numpy()

            # valid_indices = mask.sum(axis=1) > 0
            # x_long_test = x_long[valid_indices]
            # y_test = y[valid_indices]
            # label_test = label[valid_indices]

        print(f"Training size: {x_long_train.shape}, {y_train.shape}, Validation size: {x_long_val.shape}, {y_val.shape}, Test size: {x_long_test.shape}, {y_test.shape}")

        num_classes = 3  # -10 đến 10 -> 21 lớp

        # Chuyển label từ -10 đến 10 thành index từ 0 đến 20
        label_train_adj = label_train + num_classes//2
        label_val_adj = label_val + num_classes//2
        label_test_adj = label_test + num_classes//2
        
        # Nếu label_train_adj có dạng (N, k), cần flatten trước khi đếm
        flat_labels = label_train_adj.ravel()  # Chuyển về 1D nếu cần

        # Tính số lượng mẫu của mỗi class trong tập huấn luyện
        class_counts = Counter(flat_labels.tolist())

        total_samples = len(flat_labels)


        self.best_params_list = []  # Xóa danh sách tham số cũ

        # ----------------- OPTUNA TUNING CHO TỪNG OUTPUT ----------------- #
        # for i in range(self.output_size):
        for i in []:
            best_params = None
            best_score = -float("inf")

            def objective(trial):
                nonlocal best_params, best_score

                # 🔥 Optuna tối ưu trọng số class_weights 🔥
                dynamic_weight_power = trial.suggest_float("class_weight_power", 0.5, 10.0)  # Giá trị từ 0.5 đến 3.0

                class_weights = {
                    c: (total_samples / (num_classes * class_counts[c]))**dynamic_weight_power for c in class_counts
                }

                # Áp dụng trọng số tổng hợp
                sample_weights_train = np.mean(
                    np.array([[class_weights[label] for label in sample] for sample in label_train_adj]), axis=1
                )
                sample_weights_val = np.mean(
                    np.array([[class_weights[label] for label in sample] for sample in label_val_adj]), axis=1
                )

                params = {
                    "objective": "multi:softprob",
                    "num_class": num_classes,
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "lambda": trial.suggest_float("lambda", 1.0, 10.0),
                    "alpha": trial.suggest_float("alpha", 0.0, 10.0),
                    "eval_metric": "mlogloss",
                }

                dtrain = xgb.DMatrix(x_long_train, label=label_train_adj[:, i], weight=sample_weights_train)
                dval = xgb.DMatrix(x_long_val, label=label_val_adj[:, i], weight=sample_weights_val)

                model = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=10000,
                    evals=[(dval, "validation")],
                    early_stopping_rounds=15,
                    verbose_eval=False
                )

                # 🎯 Dự đoán trên tập validation
                val_preds_prob = model.predict(dval)
                val_preds = np.argmax(val_preds_prob, axis=1)

                # 🏆 Precision Score
                precision = precision_score(label_val_adj[:, i], val_preds, average="macro")

                # Lưu best params nếu precision tốt hơn
                if precision > best_score:
                    best_score = precision
                    best_params = params
                    best_params["class_weight_power"] = dynamic_weight_power  # 🔥 Lưu class_weight_power tốt nhất

                return precision  # Optuna MINIMIZE, nên phải đảo ngược

            # Chạy Optuna để tìm tham số tốt nhất cho `y[:, i]`
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=30)  # 30 lần thử nghiệm

            print(f"Best parameters for output {i}:", best_params)
            self.best_params_list.append(best_params)


        self.models = []  # Lưu nhiều mô hình

        for i in range(self.output_size):  # Tạo 1 mô hình cho từng output

            if i > 0:
                self.models.append(self.models[-1])
                continue

            # best_params = self.best_params_list[i]
            best_params =  {
                                "objective": "multi:softprob",
                                "num_class": num_classes,
                                "learning_rate": 0.1,
                                "max_depth": 6,
                                "eval_metric": ["mlogloss"],
                            }

            best_weight_power = best_params.get("class_weight_power", 0)  # Lấy class_weight_power tốt nhất

            # Tính lại trọng số với best_weight_power
            class_weights = {
                c: (total_samples / (num_classes * class_counts[c]))**best_weight_power for c in class_counts
            }

            sample_weights_train = np.mean(
                np.array([[class_weights[label] for label in sample] for sample in label_train_adj]), axis=1
            )
            sample_weights_val = np.mean(
                np.array([[class_weights[label] for label in sample] for sample in label_val_adj]), axis=1
            )
            sample_weights_test = np.mean(
                np.array([[class_weights[label] for label in sample] for sample in label_test_adj]), axis=1
            )

            idx_majority = np.where(label_train_adj[:, i] == num_classes//2)[0]  # Giả sử class 0 là nhóm chiếm ưu thế
            idx_minority = np.where(label_train_adj[:, i] != num_classes//2)[0]  # Giả sử class 1 là nhóm thiểu số

            # print (idx_majority, idx_minority)

            # Downsampling nhóm chiếm ưu thế
            n_minority = len(idx_majority)  # Số mẫu của nhóm thiểu số
            idx_majority_downsampled = np.random.choice(idx_majority, size=n_minority, replace=False)

            # Ghép lại dữ liệu sau khi downsampling
            idx_balanced = np.concatenate([idx_majority_downsampled, idx_minority])

            x_long_train_balanced = x_long_train[idx_balanced]
            label_train_balanced = label_train_adj[idx_balanced, i]
            sample_weights_train_balanced = sample_weights_train[idx_balanced]

            dtrain = xgb.DMatrix(x_long_train_balanced, label=label_train_balanced, weight=sample_weights_train_balanced)
            dval = xgb.DMatrix(x_long_val, label=label_val_adj[:, i], weight=sample_weights_val)
            dtest = xgb.DMatrix(x_long_test, label=label_test_adj[:, i], weight=sample_weights_test)

            model = xgb.train(
                params=best_params,  # Dùng bộ tham số riêng của `y[:, i]`
                dtrain=dtrain,
                num_boost_round=10000,
                evals=[(dtest, "test"), (dtrain, "train"), (dval, "validation")],
                early_stopping_rounds=15,
                verbose_eval=True
            )

            self.models.append(model)  # Lưu mô hình cho từng output riêng


        self.save_model("best_model.pkl")
        return self

    def feature_importance_selection(self, train_dataset, val_dataset, threshold=0.01, progress_file="feature_selection_progress.json"):
        """
        Select features based on their importance by removing one at a time
        and checking if performance significantly decreases. Progress is saved to a file.

        Parameters:
            train_dataset: PyTorch dataset for training
            val_dataset: PyTorch dataset for validation
            threshold: Performance drop threshold to decide feature removal
            progress_file: Filename to save progress

        Returns:
            selected_features: The final set of influential features
        """

        # Load datasets
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        for _, x_long_train, _, _, _, label_train, _ in train_loader:
            x_train = x_long_train.numpy()
            y_train = label_train.numpy()[:, 0]

        for _, x_long_val, _, _, _, label_val, _ in val_loader:
            x_val = x_long_val.numpy()
            y_val = label_val.numpy()[:, 0]

        num_classes = 3  # -10 đến 10 -> 21 lớp

        # Chuyển label từ -10 đến 10 thành index từ 0 đến 20
        y_train = y_train + num_classes//2
        y_val = y_val + num_classes//2

        # Extract feature names from dataset
        self.feature_names = train_dataset.dataset.feature_names

        selected_features = self.feature_names.copy()

        self.best_params_list = [{
                                "objective": "multi:softprob",
                                "num_class": num_classes,
                                "learning_rate": 0.1,
                                "max_depth": 6,
                                "eval_metric": ["mlogloss"],
                            }]

        # Load progress if exists
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                selected_features = progress["selected_features"]
                base_score = progress["base_score"]
                print(f"Resuming from saved progress. Current score: {base_score}")
        else:
            # Calculate initial score with all features
            dtrain_full = xgb.DMatrix(x_train, label=y_train)
            dval_full = xgb.DMatrix(x_val, label=y_val)

            full_model = xgb.train(
                params=self.best_params_list[0],
                dtrain=dtrain_full,
                num_boost_round=1000,
                evals=[(dval_full, 'validation')],
                early_stopping_rounds=15,
                verbose_eval=False
            )

            preds = np.argmax(full_model.predict(dval_full), axis=1)
            base_score = precision_score(y_val, preds, average='macro')
            print(f"Initial model score: {base_score}")

            # Save initial state
            with open(progress_file, 'w') as f:
                json.dump({"selected_features": selected_features, "base_score": base_score}, f)

        improvement = True

        while improvement and len(selected_features) > 1:
            improvement = False
            for feature in selected_features.copy():
                temp_features = [f for f in selected_features if f != feature]

                x_train_temp = x_train[:, [self.feature_names.index(f) for f in temp_features]]
                x_val_temp = x_val[:, [self.feature_names.index(f) for f in temp_features]]

                dtrain = xgb.DMatrix(x_train_temp, label=y_train)
                dval = xgb.DMatrix(x_val_temp, label=y_val)

                temp_model = xgb.train(
                    params=self.best_params_list[0],
                    dtrain=dtrain,
                    num_boost_round=1000,
                    evals=[(dval, "validation")],
                    early_stopping_rounds=15,
                    verbose_eval=False
                )

                preds = np.argmax(temp_model.predict(dval), axis=1)
                new_score = precision_score(y_val, preds, average='macro')

                print(f"Score without '{feature}': {new_score}")
                # print (type(base_score), type(new_score), type(threshold))

                if base_score - new_score <= threshold:
                    selected_features.remove(feature)
                    base_score = new_score
                    improvement = True
                    print(f"Feature '{feature}' removed. New score: {base_score}")

                    # Save progress after each removal
                    with open(progress_file, 'w') as f:
                        json.dump({"selected_features": selected_features, "base_score": base_score}, f, indent=4)

                    # break

            if not improvement:
                print("No further improvement, stopping feature selection.")

        # Final save
        with open(progress_file, 'w') as f:
            json.dump({"selected_features": selected_features, "base_score": base_score}, f, indent=4)

        return selected_features


    def plot_feature_importance(self):
        """ Plot top 100 feature importance based on the first model (assuming all models have similar importance). """
        importance = self.models[0].get_score(importance_type='weight')
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:50]  # Top 100 features

        feature_indices, scores = zip(*sorted_importance)
        feature_names = [self.feature_names[int(f[1:])] if f[1:].isdigit() else f for f in feature_indices]

        print (feature_names[:100])

        feature_names = [get_origin_feature_name(feature_name) for feature_name in feature_names]

        plt.figure(figsize=(12, 6))
        plt.barh(feature_names, scores, color='royalblue')
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature Names")
        plt.title("Top 100 XGBoost Feature Importance")
        plt.gca().invert_yaxis()

        plt.show()

        # Tạo thư mục nếu chưa tồn tại
        output_dir = "model_analysis"
        os.makedirs(output_dir, exist_ok=True)

         # Tạo tên file với timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{timestamp}_feature_importance.png")

        # Lưu hình ảnh
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Feature importance plot saved to {output_path}")


    def evaluate(self, dataset, test_indices):
        X = xgb.DMatrix(dataset.X_long_term[test_indices])
        label = dataset.labels[test_indices]

        # Lặp qua tất cả các model trong self.models
        for i, model in enumerate(self.models[:1]):
            print(f"Đang đánh giá model thứ {i}:")
            probs = model.predict(X)  # Lấy xác suất cho mỗi lớp từ model hiện tại

            # print("Kích thước X_long_term:", dataset.X_long_term[test_indices].shape)
            # print("Kích thước label:", label.shape)
            # print("Xác suất dự đoán:", probs)
            # print("Kích thước xác suất:", probs.shape)

            # Chuyển đổi xác suất thành nhãn dự đoán:
            # Giả sử nhãn ban đầu được mã hóa từ -50 đến 50 (với 101 lớp)
            preds = np.argmax(probs, axis=1) - 1

            # Trích xuất cột label thứ i, có shape (296,)
            true_labels = label[:, i]

            # print("Nhãn dự đoán:", preds)
            # print("Nhãn thực:", true_labels)
            print("Báo cáo phân loại:")
            print(classification_report(true_labels, preds))

    def evaluate_high_confidence(self, dataset, test_indices):
        X = xgb.DMatrix(dataset.X_long_term[test_indices])
        label = dataset.labels[test_indices]
        percentages = dataset.percentages[test_indices]
        close_prices = dataset.close_prices[test_indices]

        total_gains = 0

        for i, model in enumerate(self.models[:1]):
            print(f"Đang đánh giá model thứ {i}:")
            probs = model.predict(X)

            max_probs = np.max(probs, axis=1)
            preds = np.argmax(probs, axis=1) - 1  # Điều chỉnh về khoảng -50 đến 50

            true_labels = label[:, i]

            # true_percentages = percentages[:, i]

            confident_mask = max_probs >= 0.55
            confident_preds = preds[confident_mask]
            confident_true_labels = true_labels[confident_mask]
            # confident_percentages = true_percentages[confident_mask]
            confident_close_prices = close_prices[confident_mask]

            # Chỉ lấy những dự đoán là 1 (long) hoặc -1 (short), bỏ qua 0
            trade_mask = (confident_preds != 0)
            confident_preds = confident_preds[trade_mask]
            confident_true_labels = confident_true_labels[trade_mask]
            # confident_percentages = confident_percentages[trade_mask]
            confident_close_prices = confident_close_prices[trade_mask]

            print(f"{len(confident_preds)} predictions (long/short) with confidence >= 55%")


            if len(confident_preds) == 0:
                print("Không có dự đoán nào đạt mức tự tin >= 50%")
                continue

            # Tính toán lợi nhuận với long/short
            # gains = np.where(
            #     confident_preds == 1,  # Nếu long
            #     confident_close_prices * confident_percentages - 1,
            #     - confident_close_prices * confident_percentages - 1  # Nếu short
            # )

            # total_gains += np.sum(gains)

            # Đếm số lượng prediction cho từng class
            count_class_neg = np.sum(confident_preds == -1)
            count_class_pos = np.sum(confident_preds == 1)

            print(f"Số lượng prediction class -1 với confidence >= 70%: {count_class_neg}")
            print(f"Số lượng prediction class 1 với confidence >= 70%: {count_class_pos}")

            precisions, recalls = [], []
            for cls in [-1, 1]:
                total_cls = np.sum(true_labels == cls)

                recall_mask = (confident_preds == cls) & (confident_true_labels == cls)
                recall = recall_mask.sum() / total_cls if total_cls > 0 else 0

                precision_mask = confident_preds == cls
                precision = (recall_mask.sum() / precision_mask.sum()) if precision_mask.sum() > 0 else 0

                print(f"Class {cls} - Precision: {precision:.4f}, Recall: {recall:.4f}")

            # Tính precision, recall macro và micro
            precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
                confident_true_labels, confident_preds, labels=[-1, 1], average='macro', zero_division=0
            )

            precision_micro, recall_micro, _, _ = precision_recall_fscore_support(
                confident_true_labels, confident_preds, labels=[-1, 1], average='micro', zero_division=0
            )

            print(f"Macro Precision: {precision_macro:.4f}, Macro Recall: {recall_macro:.4f}")
            print(f"Micro Precision: {precision_micro:.4f}, Micro Recall: {recall_micro:.4f}")

    def evaluate_high_confidence_3_classes(self, dataset, test_indices):
        X = xgb.DMatrix(dataset.X_long_term[test_indices])
        label = dataset.labels[test_indices]

        for i, model in enumerate(self.models[:1]):
            print(f"Đang đánh giá model thứ {i}:")
            probs = model.predict(X)  # Xác suất cho mỗi lớp

            # Lấy lớp có xác suất cao nhất và xác suất tương ứng
            max_probs = np.max(probs, axis=1)
            preds = np.argmax(probs, axis=1) - 100  # Điều chỉnh về khoảng -50 đến 50

            # Trích xuất nhãn thực tế
            true_labels = label[:, i]

            # Chuyển đổi thành 3 nhóm {-1, 0, 1}
            preds_grouped = np.where(preds > 0, 1, np.where(preds < 0, -1, 0))
            true_labels_grouped = np.where(true_labels > 0, 1, np.where(true_labels < 0, -1, 0))

            # Chỉ giữ lại các dự đoán mà mô hình tự tin >= 50%
            confident_mask = max_probs >= 0.5
            confident_preds = preds_grouped[confident_mask]
            confident_true_labels = true_labels_grouped[confident_mask]

            if len(confident_preds) == 0:
                print("Không có dự đoán nào đạt mức tự tin >= 50%")
                continue

            # Tính precision, recall trên tập dữ liệu đã lọc với 3 nhóm
            precision, recall, _, _ = precision_recall_fscore_support(
                confident_true_labels, confident_preds, average='weighted'
            )

            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")


    def forward(self, x_short, x_long):
        """
        Perform inference with classification models.
        """
        x_long = x_long.cpu().numpy()
        dtest = xgb.DMatrix(x_long)

        # Lưu danh sách kết quả dự đoán từ các mô hình
        predicted_labels_list = []
        max_probs_list = []

        for model in self.models[:1]:  # Duyệt từng mô hình (tương ứng với từng output)
            probs = model.predict(dtest)  # Predict xác suất (N, num_classes)
            predicted_labels = np.argmax(probs, axis=1) - 1  # Chuyển về khoảng [-10, 10]

            # predicted_labels = label_to_percent(predicted_labels)
            predicted_labels = predicted_labels * 0.5

            max_probs = np.max(probs, axis=1)  # Xác suất cao nhất của mỗi mẫu

            predicted_labels_list.append(predicted_labels)
            max_probs_list.append(max_probs)

        # Chuyển danh sách thành numpy array với shape (batch_size, output_size)
        predicted_labels_array = np.column_stack(predicted_labels_list)
        max_probs_array = np.column_stack(max_probs_list)

        # Chuyển về tensor PyTorch
        predictions = torch.tensor(predicted_labels_array, dtype=torch.float32)
        max_probs_tensor = torch.tensor(max_probs_array, dtype=torch.float32)

        return predictions, predictions, max_probs_tensor  # Trả về kết quả dự đoán & xác suất cao nhất
    
    def forward_2(self, x_short, x_long, top_k=3):
        """
        Perform inference with classification models and return only top-k predictions.
        """
        x_long = x_long.cpu().numpy()
        dtest = xgb.DMatrix(x_long)

        top_k_labels_list = []
        top_k_probs_list = []

        for model in self.models:  # Duyệt từng mô hình (tương ứng với từng output)
            probs = model.predict(dtest)  # Predict xác suất (N, num_classes)

            # Lấy top-k nhãn có xác suất cao nhất
            top_k_indices = np.argsort(probs, axis=1)[:, -top_k:]  # (N, top_k)
            top_k_probs = np.take_along_axis(probs, top_k_indices, axis=1)  # (N, top_k)

            # Chuyển về khoảng [-10, 10]
            top_k_indices -= 100
            top_k_indices = label_to_percent(top_k_indices)

            top_k_labels_list.append(top_k_indices)
            top_k_probs_list.append(top_k_probs)

        # Chuyển danh sách thành numpy array với shape (batch_size, output_size, top_k)
        top_k_labels_array = np.stack(top_k_labels_list, axis=1)  # (batch_size, output_size, top_k)
        top_k_probs_array = np.stack(top_k_probs_list, axis=1)  # (batch_size, output_size, top_k)

        # Hoán vị để chiều đầu tiên là top_k: (top_k, batch_size, output_size)
        top_k_labels_array = np.transpose(top_k_labels_array, (2, 0, 1))
        top_k_probs_array = np.transpose(top_k_probs_array, (2, 0, 1))

        # Chuyển về tensor PyTorch
        top_k_labels_tensor = torch.tensor(top_k_labels_array, dtype=torch.float32)
        top_k_probs_tensor = torch.tensor(top_k_probs_array, dtype=torch.float32)

        return top_k_labels_tensor, top_k_labels_tensor, top_k_probs_tensor

class LightGBMClassifier(PriceClassifier):
    def fit(self, train_dataset, val_dataset, test_dataset):
        self.prepare_data(train_dataset, val_dataset, test_dataset)

        self.models = []  # Lưu nhiều mô hình

        for i in range(self.output_size):  # Tạo 1 mô hình cho từng output

            if i > 0:
                self.models.append(self.models[-1])
                continue

            dtrain = lgb.Dataset(self.x_long_train, label=self.label_train[:, i])
            dval = lgb.Dataset(self.x_long_val, label=self.label_val[:, i], reference=dtrain)
            dtest = lgb.Dataset(self.x_long_test, label=self.label_test[:, i], reference=dtrain)

            params = {
                "objective": "multiclass",
                "num_class": 3,  # 21 classes from -10 to 10
                "metric": "multi_logloss",
                "learning_rate": 0.05,
                "max_depth": 7,
                "num_leaves": 31,
                "min_data_in_leaf": 20,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "lambda_l1": 0.1,
                "lambda_l2": 0.1,
                "verbose": -1,
            }

            model = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=10000,
                valid_sets=[dtest, dtrain, dval],
                valid_names=["test", "train", "validation"],
                early_stopping_rounds=15,
                verbose_eval=True
            )

            self.models.append(model)  # Lưu mô hình cho từng output riêng


        self.save_model("best_lgm_model.pkl")
        return self

    def evaluate(self, dataset, test_indices):
        X = dataset.X_long_term[test_indices]
        label = dataset.labels[test_indices]

        # Lặp qua tất cả các model trong self.models
        for i, model in enumerate(self.models[:1]):
            print(f"Đang đánh giá model thứ {i}:")
            probs = model.predict(X)  # Lấy xác suất cho mỗi lớp từ model hiện tại

            # print("Kích thước X_long_term:", dataset.X_long_term[test_indices].shape)
            # print("Kích thước label:", label.shape)
            # print("Xác suất dự đoán:", probs)
            # print("Kích thước xác suất:", probs.shape)

            # Chuyển đổi xác suất thành nhãn dự đoán:
            # Giả sử nhãn ban đầu được mã hóa từ -50 đến 50 (với 101 lớp)
            preds = np.argmax(probs, axis=1) - 1

            # Trích xuất cột label thứ i, có shape (296,)
            true_labels = label[:, i]

            # print("Nhãn dự đoán:", preds)
            # print("Nhãn thực:", true_labels)
            print("Báo cáo phân loại:")
            print(classification_report(true_labels, preds))

    def evaluate_high_confidence(self, dataset, test_indices):
        X = dataset.X_long_term[test_indices]
        label = dataset.labels[test_indices]

        for i, model in enumerate(self.models[:1]):
            print(f"Đang đánh giá model thứ {i}:")
            probs = model.predict(X)

            max_probs = np.max(probs, axis=1)
            preds = np.argmax(probs, axis=1) - 1  # Điều chỉnh về khoảng -50 đến 50

            true_labels = label[:, i]

            confident_mask = max_probs >= 0.5
            confident_preds = preds[confident_mask]
            confident_true_labels = true_labels[confident_mask]

            print(f"{len(confident_preds)} predictions with confidence >= 50%")

            if len(confident_preds) == 0:
                print("Không có dự đoán nào đạt mức tự tin >= 50%")
                continue

            # Đếm số lượng prediction cho từng class
            count_class_neg = np.sum(confident_preds == -1)
            count_class_pos = np.sum(confident_preds == 1)

            print(f"Số lượng prediction class -1 với confidence >= 70%: {count_class_neg}")
            print(f"Số lượng prediction class 1 với confidence >= 70%: {count_class_pos}")

            precisions, recalls = [], []
            for cls in [-1, 1]:
                total_cls = np.sum(true_labels == cls)

                recall_mask = (confident_preds == cls) & (confident_true_labels == cls)
                recall = recall_mask.sum() / total_cls if total_cls > 0 else 0

                precision_mask = confident_preds == cls
                precision = (recall_mask.sum() / precision_mask.sum()) if precision_mask.sum() > 0 else 0

                print(f"Class {cls} - Precision: {precision:.4f}, Recall: {recall:.4f}")

            # Tính precision, recall macro và micro
            precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
                confident_true_labels, confident_preds, labels=[-1, 1], average='macro', zero_division=0
            )

            precision_micro, recall_micro, _, _ = precision_recall_fscore_support(
                confident_true_labels, confident_preds, labels=[-1, 1], average='micro', zero_division=0
            )

            print(f"Macro Precision: {precision_macro:.4f}, Macro Recall: {recall_macro:.4f}")
            print(f"Micro Precision: {precision_micro:.4f}, Micro Recall: {recall_micro:.4f}")


class EnsembleClassifier(PriceClassifier):
    def __init__(self, output_size, ensemble_method="stacking"):
        super().__init__(output_size)
        self.xgb_models = []
        self.lgb_models = []
        self.cat_models = []
        self.meta_models = []

    def fit(self, train_dataset, val_dataset, test_dataset):
        self.prepare_data(train_dataset, val_dataset, test_dataset)

        self.xgb_models = []
        self.lgb_models = []
        self.cat_models = []
        self.meta_models = []

        meta_features_train = []
        meta_features_val = []

        for i in range(self.output_size):
            dtrain = xgb.DMatrix(self.x_long_train, label=self.label_train[:, i])
            dval = xgb.DMatrix(self.x_long_val, label=self.label_val[:, i])

            xgb_params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "learning_rate": 0.1,
                "max_depth": 6,
            }
            xgb_model = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=500, evals=[(dval, "validation")], early_stopping_rounds=15, verbose_eval=True)
            self.xgb_models.append(xgb_model)

            lgb_train = lgb.Dataset(self.x_long_train, label=self.label_train[:, i])
            lgb_val = lgb.Dataset(self.x_long_val, label=self.label_val[:, i], reference=lgb_train)

            lgb_params = {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "learning_rate": 0.05
            }
            from lightgbm import early_stopping
            lgb_model = lgb.train(params=lgb_params, train_set=lgb_train, num_boost_round=500, valid_sets=[lgb_val], callbacks=[early_stopping(stopping_rounds=15, verbose=True)])
            self.lgb_models.append(lgb_model)

            cat_model = cb.CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, loss_function='MultiClass', verbose=1)
            cat_model.fit(self.x_long_train, self.label_train[:, i], eval_set=(self.x_long_val, self.label_val[:, i]), early_stopping_rounds=15)
            self.cat_models.append(cat_model)

            # Collect meta-features for stacking
            xgb_train_preds = xgb_model.predict(dtrain)
            lgb_train_preds = lgb_model.predict(self.x_long_train)
            cat_train_preds = cat_model.predict_proba(self.x_long_train)
            meta_features_train.append(np.hstack([xgb_train_preds, lgb_train_preds, cat_train_preds]))

            xgb_val_preds = xgb_model.predict(dval)
            lgb_val_preds = lgb_model.predict(self.x_long_val)
            cat_val_preds = cat_model.predict_proba(self.x_long_val)
            meta_features_val.append(np.hstack([xgb_val_preds, lgb_val_preds, cat_val_preds]))

            break

        meta_features_train = np.hstack(meta_features_train)
        meta_features_val = np.hstack(meta_features_val)

        for i in range(self.output_size):
            meta_model = xgb.XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", learning_rate=0.05, max_depth=5, n_estimators=200,                              early_stopping_rounds=15)
            meta_model.fit(meta_features_train, self.label_train[:, i], eval_set=[(meta_features_val, self.label_val[:, i])], verbose=True)
            self.meta_models.append(meta_model)

            break

        self.save_model("best_ensemble_model.pkl")
        return self


    def evaluate(self, dataset, test_indices):
        X = dataset.X_long_term[test_indices]
        label = dataset.labels[test_indices]

        xgb_dtest = xgb.DMatrix(X)
        meta_features_test = []
        for i in range(self.output_size):
            xgb_preds = self.xgb_models[i].predict(xgb_dtest)
            lgb_preds = self.lgb_models[i].predict(X)
            cat_preds = self.cat_models[i].predict_proba(X)

            meta_features_test.append(np.hstack([xgb_preds, lgb_preds, cat_preds]))

            break

        meta_features_test = np.hstack(meta_features_test)
        final_preds = np.column_stack([model.predict(meta_features_test) - 1 for model in self.meta_models])

        true_labels = label[:, :1]

        # print (final_preds)
        #
        # print("XGB predictions:", np.unique(np.argmax(xgb_preds, axis=1)))
        # print("LGB predictions:", np.unique(np.argmax(lgb_preds, axis=1)))
        # print("CAT predictions:", np.unique(np.argmax(cat_preds, axis=1)))
        # print("Meta predictions:", np.unique(final_preds))
        # print("groud truth:", np.unique(final_preds))
        # exit()

        print(classification_report(true_labels.ravel(), final_preds.ravel()))

    
    def evaluate_high_confidence(self, dataset, test_indices):
        X = dataset.X_long_term[test_indices]
        label = dataset.labels[test_indices]

        xgb_dtest = xgb.DMatrix(X)
        meta_features_test = []
        for i in range(self.output_size):
            xgb_preds = self.xgb_models[i].predict(xgb_dtest)
            lgb_preds = self.lgb_models[i].predict(X)
            cat_preds = self.cat_models[i].predict_proba(X)
            meta_features_test.append(np.hstack([xgb_preds, lgb_preds, cat_preds]))

            break

        meta_features_test = np.hstack(meta_features_test)

        for i, model in enumerate(self.meta_models[:1]):
            print(f"Đang đánh giá model thứ {i}:")
            probs = model.predict_proba(meta_features_test)

            max_probs = np.max(probs, axis=1)
            preds = np.argmax(probs, axis=1) - 1  # Điều chỉnh về khoảng -1, 0, 1

            true_labels = label[:, i]

            confident_mask = max_probs >= 0.47
            confident_preds = preds[confident_mask]
            confident_true_labels = true_labels[confident_mask]

            print(f"{len(confident_preds)} predictions with confidence >= 50%")

            if len(confident_preds) == 0:
                print("Không có dự đoán nào đạt mức tự tin >= 50%")
                continue

            count_class_neg = np.sum(confident_preds == -1)
            count_class_pos = np.sum(confident_preds == 1)

            print(f"Số lượng prediction class -1: {count_class_neg}")
            print(f"Số lượng prediction class 1: {count_class_pos}")

            for cls in [-1, 1]:
                total_cls = np.sum(true_labels == cls)
                recall_mask = (confident_preds == cls) & (confident_true_labels == cls)
                recall = recall_mask.sum() / total_cls if total_cls > 0 else 0
                precision_mask = confident_preds == cls
                precision = (recall_mask.sum() / precision_mask.sum()) if precision_mask.sum() > 0 else 0
                print(f"Class {cls} - Precision: {precision:.4f}, Recall: {recall:.4f}")

            precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
                confident_true_labels, confident_preds, labels=[-1, 1], average='macro', zero_division=0
            )
            precision_micro, recall_micro, _, _ = precision_recall_fscore_support(
                confident_true_labels, confident_preds, labels=[-1, 1], average='micro', zero_division=0
            )

            print(f"Macro Precision: {precision_macro:.4f}, Macro Recall: {recall_macro:.4f}")
            print(f"Micro Precision: {precision_micro:.4f}, Micro Recall: {recall_micro:.4f}")

