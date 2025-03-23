import argparse
import os, sys

# Get the absolute path of the dataset directory
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
sys.path.append(dataset_path)

from dataset import TARGET_LENGTH, get_train_val_test
from regression_model import *
from classification_model import *
from svm import *
def train_model(model_name):
    ##############################################
    # 1. Load Data
    ##############################################
    _, train_dataset, val_dataset, test_dataset, _, _, _ = get_train_val_test()
    # Mô hình
    model_classes = {
        "lstm": LSTMQuantileRegressor,
        "gru": GRUQuantileRegressor,
        "cnn": CNNQuantileRegressor,
        "lr": LinearQuantileRegressor,
        "rnn": RNNQuantileRegressor,
        "dfae": DeepFeedforwardRegressor,
        "xgboost_classifier": XGBoostClassifier,
        "svm": SVMClassifier
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Invalid model name '{model_name}'. Choose from lstm, gru, cnn, or lr.")
    
    model = model_classes[model_name](output_size=TARGET_LENGTH)
    
    # Train 
    model.fit(train_dataset, val_dataset, test_dataset)

# model = LSTMQuantileRegressor(output_size=1)
# model = XGBoostClassifier(output_size=TARGET_LENGTH)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Train a quantile regression model.")
    # parser.add_argument("--model", type=str, required=True,help="Model type:lstm, gru, cnn, rnn , dfae")
    # args = parser.parse_args()
    model_name = "svm"
    train_model(model_name)

