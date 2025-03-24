import shutil

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import copy
import os, sys

# Get the absolute path of the dataset directory
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
sys.path.append(dataset_path)

from dataset import TARGET_LENGTH, get_train_val_test
from features_cal_tg_and_step import label_to_percent

from regression_model import *
from classification_model import *
from svm import *
from bigru_lstm import BiGRU_LSTM_Clasiifier
##############################################
# 1. Load the trained model
##############################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = CNNQuantileRegressor.load_model("best_model.pkl").to(device)
#
# model = XGBoostClassificationModel.load_model("best_classification_model.pkl").to(device)

# model = LightGBMClassifier.load_model("best_lgb_model_2.pkl").to(device)
# model = XGBoostClassifier.load_model("best_model.pkl").to(device)

model = BiGRU_LSTM_Clasiifier()
model.load_model("exps/bigru_lstm_all_models.pt")
# model = EnsembleClassifier.load_model("best_ensemble_model.pkl").to(device)
# model.eval()

# model.plot_feature_importance();
#
# exit()

##############################################
# 2. Load the test dataset
##############################################
dataset, train_dataset, val_dataset, test_dataset, train_indices, val_indices, test_indices = get_train_val_test()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# # Print classification report:
# # model.evaluate(dataset, train_indices)
# model.evaluate(dataset, val_indices)
# model.evaluate(dataset, test_indices)
# model.evaluate_high_confidence(dataset, val_indices)
# model.evaluate_high_confidence(dataset, test_indices)
model.evaluate(val_dataset)
model.evaluate(test_dataset)
exit()

# # Print classification report:
# # model.evaluate(dataset, train_indices)
# model.evaluate_high_confidence(dataset, val_indices)
# model.evaluate_high_confidence(dataset, test_indices)
# model.evaluate_high_confidence_3_classes(dataset, test_indices)
# exit()

##############################################
# 3. Create output folder for images
##############################################
output_dir = "./smoothed_test_predictions_with_branches"

# Xóa thư mục cũ nếu tồn tại
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir, exist_ok=True)

##############################################
# 4. Test and visualize results with branching
##############################################
# Store predictions for smoothing
predictions_per_day = {}
step_counters = {}

top_k = 1

for idx, (x_short, x_long, y_true, mask, percent, label, close) in enumerate(test_loader):
    x_short, x_long, label, mask = (
        x_short.to(device),
        x_long.to(device),
        label.to(device),
        mask.to(device),
    )
    # print ("x_short:", x_short)

    # Get the sample date
    sample_date = dataset.sample_dates[test_indices[idx]]

    # Initialize if it's the first time seeing this date
    if sample_date not in predictions_per_day:
        predictions_per_day[sample_date] = {
            "pred_5": np.full((top_k, TARGET_LENGTH), np.nan),
            "pred_50": np.full((top_k, TARGET_LENGTH), np.nan),
            "probs_50": np.full((top_k, TARGET_LENGTH), np.nan),
            "ground_truth": np.full(TARGET_LENGTH, np.nan),
            "mask": np.zeros(TARGET_LENGTH),
            "first_close": close.cpu().numpy()[0]
        }

        step_counters[sample_date] = 1  # Initialize step counter

    # Chuyển từ tensor sang numpy
    label = y_true.cpu().numpy()

    # # Áp dụng label_to_percent cho từng phần tử
    # label = np.vectorize(label_to_percent)(label)


    # Make predictions
    with torch.no_grad():
        pred_5, pred_50, probs_50 = model.forward_2(x_short, x_long, top_k)

        print ("pred_50:", pred_50)
        print ("probs_50", probs_50)
        print ("y_true", label)

    # Convert to numpy
    y_true_np = label.flatten()
    pred_5_np = pred_5.cpu().numpy().squeeze(1)  # Shape (top_k, output_size)
    pred_50_np = pred_50.cpu().numpy().squeeze(1)  # Shape (top_k, output_size)
    probs_50_np = probs_50.cpu().numpy().squeeze(1)  # Shape (top_k, output_size)
    mask_np = mask.cpu().numpy().flatten()

    # predictions_per_day[sample_date]["ground_truth"] = y_true_np


    # Apply mask (ignore padding)
    valid_indices = np.arange(len(mask_np))
    # print (len(valid_indices))
    # exit()
    # valid_indices = np.where(mask_np == 1)[0]

    # Save previous predictions for branching visualization
    prev_pred_50 = copy.deepcopy(predictions_per_day[sample_date]["pred_50"])

    # Update predictions for new valid time steps
    # Update predictions ONLY for new valid time steps
    current_step = step_counters[sample_date] - 1  # Adjusting for 0-based indexing

    pred_50_np = close.cpu().numpy()[0] * (pred_50_np/100 + 1)

    for i in valid_indices:
        index = i + current_step
        if index >= TARGET_LENGTH:  # Giới hạn chỉ số
            continue  # Bỏ qua nếu vượt quá giới hạn

        # ✅ Cập nhật tất cả `top_k` đường dự đoán
        for k in range(top_k):
            predictions_per_day[sample_date]["pred_5"][k, index] = pred_5_np[k, i]
            predictions_per_day[sample_date]["pred_50"][k, index] = pred_50_np[k, i]
            predictions_per_day[sample_date]["probs_50"][k, index] = probs_50_np[k, i]

        # ✅ Cập nhật ground truth (vẫn chỉ có một giá trị)
        predictions_per_day[sample_date]["ground_truth"][index] = y_true_np[i]
        predictions_per_day[sample_date]["mask"][index] = 1  # Mark as valid


    # ✅ Tại các timestep cũ, cập nhật lại bằng ground truth
    print ("current_step", current_step)
    # ✅ Cập nhật cho tất cả top_k đường dự đoán
    for k in range(top_k):
        predictions_per_day[sample_date]["pred_5"][k, :current_step] = predictions_per_day[sample_date]["ground_truth"][:current_step]
        predictions_per_day[sample_date]["pred_50"][k, :current_step] = predictions_per_day[sample_date]["ground_truth"][:current_step]


    # Plotting
    timestamps = np.arange(TARGET_LENGTH)
    pred_5_smooth = predictions_per_day[sample_date]["pred_5"]
    pred_50_smooth = predictions_per_day[sample_date]["pred_50"]
    probs_50_smooth = predictions_per_day[sample_date]["probs_50"]
    ground_truth = predictions_per_day[sample_date]["ground_truth"]
    valid_mask = predictions_per_day[sample_date]["mask"]


    # Create the plot
    plt.figure(figsize=(12, 6))

    # Chèn giá trị 0 vào đầu dãy
    # ground_truth = np.insert(np.nancumsum(ground_truth), 0, 0)
    # prev_pred_50 = np.insert(np.nancumsum(prev_pred_50), 0, 0)
    # pred_50_smooth = np.insert(np.nancumsum(pred_50_smooth), 0, 0)
    # pred_5_smooth = np.insert(np.nancumsum(pred_5_smooth), 0, 0)
    # probs_50_smooth = np.insert(probs_50_smooth, 0, 0)

    ground_truth = np.insert(ground_truth, 0, predictions_per_day[sample_date]["first_close"])  # Shape (TARGET_LENGTH + 1,)

    prev_pred_50 = np.insert(prev_pred_50, 0, predictions_per_day[sample_date]["first_close"], axis=1)  # Shape (top_k, TARGET_LENGTH + 1)
    pred_50_smooth = np.insert(pred_50_smooth, 0, predictions_per_day[sample_date]["first_close"], axis=1)  # Shape (top_k, TARGET_LENGTH + 1)
    pred_5_smooth = np.insert(pred_5_smooth, 0, predictions_per_day[sample_date]["first_close"], axis=1)  # Shape (top_k, TARGET_LENGTH + 1)
    probs_50_smooth = np.insert(probs_50_smooth, 0, 0, axis=1)  # Shape (top_k, TARGET_LENGTH + 1)


    # Cập nhật timestamps để có thêm điểm đầu tiên (0)
    timestamps = np.arange(TARGET_LENGTH + 1)
    valid_indices = valid_indices + 1  # Dịch chỉ số lên 1 do đã thêm 1 phần tử vào đầu dãy

    # Plot Ground Truth (fixed for all steps)
    plt.plot(timestamps, ground_truth, label="Ground Truth", marker="o", linestyle="-", color="black")

    # # Plot Previous Predictions (before current update)
    # plt.plot(timestamps, prev_pred_50, label="Previous Predictions", linestyle="--", color="blue", alpha=0.6)

    # Chỉ highlight những điểm sau current_step
    valid_indices_new = valid_indices[valid_indices > current_step]

    # # Highlight New Predictions (branch points)
    # plt.scatter(valid_indices_new, pred_50_smooth[valid_indices_new],
    #             color="orange", label="New Predictions", marker="D", s=70)

    # Tạo màu sắc khác nhau cho từng đường dự đoán
    colors = plt.cm.viridis(np.linspace(0.5, 1, top_k))

    # ✅ Lặp qua từng `top_k` để highlight đúng màu và đường
    for k in range(top_k):
        # Highlight New Predictions (branch points)
        plt.scatter(valid_indices_new, pred_50_smooth[k, valid_indices_new],
                    color=colors[k], label=f"New Prediction {k+1}", marker="o", s=50)

        # Chỉ lấy xác suất tại các điểm mới sau `current_step`
        valid_probs = probs_50_smooth[k, valid_indices_new]

        # Vẽ giá trị xác suất lên biểu đồ
        for i, prob in zip(valid_indices_new, valid_probs):
            plt.text(i, pred_50_smooth[k, i] + 0.04, f"{prob:.2f}",
                     color=colors[k], fontsize=10, ha="center", weight="bold")

    # # Draw branches from old to new predictions
    # for i in valid_indices:
    #     if not np.isnan(prev_pred_50[i]):
    #         plt.plot([i, i], [prev_pred_50[i], pred_50_smooth[i]], color="orange", linestyle="--")

    # ✅ Lặp qua từng `top_k` để vẽ đúng dữ liệu
    for k in range(top_k):
        plt.plot(timestamps, pred_50_smooth[k],
                 label=f"Smoothed Prediction {k+1}", linestyle="-", color=colors[k])


    # # Fill the confidence interval (5% - 95%)
    # plt.fill_between(timestamps, pred_5_smooth, pred_95_smooth, color="gray", alpha=0.2, label="Confidence Interval")

    plt.xlabel("Time Steps")
    plt.ylabel("Log Return")
    plt.title(f"Predictions with Branching - {sample_date} (Step {step_counters[sample_date]})")
    plt.legend()
    plt.grid()

    # Save the figure
    image_path = os.path.join(output_dir, f"predictions_{sample_date}_{step_counters[sample_date]}.png")
    plt.savefig(image_path)
    plt.close()

    print(f"Saved: {image_path}")

    # Increment step counter
    step_counters[sample_date] += 1

    # exit()

print("All smoothed test images saved successfully.")

