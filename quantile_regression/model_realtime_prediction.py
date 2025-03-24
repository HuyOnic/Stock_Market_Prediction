import time
import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine, text
import datetime
from regression_model import CNNQuantileRegressor
from classification_model import XGBoostClassifier
from database_utils import get_engine
import os, sys
import joblib
import json
import argparse
from quantile_regression.bigru_lstm import BiGRU_LSTM_Clasiifier
# Get dataset path
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
sys.path.append(dataset_path)
from dataset import FinancialDataset, load_data, STEP_MINUTES, SequenceFinancialDataset
#ALl features below will be use for Huy's model
SELECTED_LONG_FEATURES = ['f45', 'f48', 'f47', 'f39', 'f40', 'f41', 'f42', 'f43', 'f6', 
                          'f7', 'f8', 'f9', 'f10', 'f13', 'f14', 'f15', 
                          'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 
                          'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 
                          'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 
                          'f38', 'f176', 'f177', 'f178', 'f179', 'f180', 
                          'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 
                          'f56', 'f57', 'f58', 'f59', 'f60', 'f61', 'f62', 
                          'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69', 
                          'f70', 'f71', 'f72', 'f73', 'f74', 'f75', 'f76', 
                          'f77', 'f78', 'f79', 'f80', 'f81', 'f82', 'f83', 
                          'f84', 'f85', 'f86', 'f87', 'f88', 'f89', 'f90', 
                          'f91', 'f92', 'f93', 'f94', 'f95', 'f96', 'f97', 
                          'f98', 'f99', 'f100', 'f101', 'f102', 'f103', 'f104', 'f105', 'f106', 'f107', 'f108', 'f109', 'f110', 'f111', 'f114', 'f115', 'f116', 'f117', 'f118', 'f119', 'f120', 'f121', 'f122', 'f123', 'f124', 'f125', 'f126', 'f127', 'f128', 'f129', 'f130', 'f131', 'f132', 'f133', 'f134', 'f135', 'f136', 'f137', 'f138', 'f139', 'f140', 'f141', 'f142', 'f143', 'f144', 'f145', 'f146', 'f147', 'f148', 'f149', 'f150', 'f151', 'f154', 'f155', 'f156', 'f157', 'f158', 'f159', 'f160', 'f161', 'f162', 'f163', 'f164', 'f165', 'f166', 'f167', 'f168', 'f169', 'f170', 'f171', 'f172', 'f173']
##############################################
# Load Models
##############################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_configs = {
    # "xgboost-clss": {
    #     "path": "checkpoints/xgboost_cls_20250302.pkl",
    #     "class": XGBoostClassifier,
    #     "scaler": "checkpoints/scaler_20250302.pkl"
    # },
    # "xgboost_cls_20250304_all_samples_down_0": {
    #     "path": "checkpoints/xgboost_cls_20250304_all_samples_down_0.pkl",
    #     "class": XGBoostClassifier,
    #     "scaler": "checkpoints/scaler_20250304_all_samples_down_0.pkl"
    # }
    # "xgb_3": {
    #     "path": "best_model.pkl",
    #     "class": XGBoostClassifier,
    #     "scaler": "checkpoints/scaler_20250304_all_samples_down_0.pkl"
    # }
    "bigru_lstm": {
        "path": "exps/bigru_lstm_all_models.pt",
        "class": BiGRU_LSTM_Clasiifier,
        "scaler": "min_max_scaler.pkl"
    }
}
models = {
    model_name: config["class"]
    for model_name, config in model_configs.items()
}

# seq_models = {
#     model_name: config["class"].load_model(config)
#     for model_name, config in model_configs.items()
# }
# for model in seq_models.values():
#     if isinstance(model, list):
#         for child_model in model: 
#             child_model.eval()
#         continue
##############################################
# Database Connection
##############################################
engine = get_engine()
CHECK_WINDOW_MINUTES = 5

##############################################
# Realtime Prediction Loop
##############################################

def get_unpredicted_data(start_date, end_date, model_name):
    """ Fetch dữ liệu mới chưa được dự đoán từ database trong khoảng ngày"""
    df = load_data(load_from_db=False, table_name="der_1m_feature",
                   eod_table="der_1d_feature", der_1m_table="der_1m",
                   save_to_file=False, start_date=start_date, end_date=end_date)

    query = f'''
        SELECT trade_date, "time"
        FROM public.der_model_output
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}' AND model_name = '{model_name}'
    '''

    with engine.connect() as conn:
        predicted_df = pd.read_sql(text(query), conn)

    df = df.merge(predicted_df, on=["trade_date", "time"], how="left", indicator=True)
    df = df[df["_merge"] == "left_only"].drop(columns=["_merge"])

    return df

def predict_and_save(df, model_name, model):
    """ Dự đoán trên dữ liệu mới và lưu kết quả vào database """
    if df.empty:
        print(f"✅ No new data to predict for {model_name}.")
        return

    df.fillna(0, inplace=True)
    dataset = FinancialDataset(df, window_size=30)
    if len(dataset) == 0:
        print(f"✅ No new data to predict for {model_name}.")
        return

    # scaler = joblib.load(model_configs[model_name]["scaler"])
    # dataset.scale_data(scaler)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    results = []
    for idx, (x_short, x_long, _, _, _, _, _) in enumerate(dataloader):
        x_short, x_long = x_short.to(device), x_long.to(device)

         # trade_date = int(df.iloc[idx]['trade_date'])
        trade_date = int(dataset.trade_dates[idx])

        timestamp = pd.to_datetime(dataset.sample_times[idx])
        close = dataset.close_prices[idx]
        time_frame = 15
        model.eval()
        with torch.no_grad():
            _, preds, probs = model(x_short, x_long)

            print(f"🔍 Prediction at {timestamp}: {preds.cpu().numpy()[0]} (Probability: {probs.cpu().numpy()[0]})")

            if probs.cpu().numpy()[0][0] < 0.55:
                continue

            CUTOFF_TIME = datetime.datetime.strptime("07:45", "%H:%M").time()

            predictions = [
                (
                    final_timestamp,
                    close * (100 + pred) / 100,
                    prob
                )
                for i, (pred, prob) in enumerate(zip(preds.cpu().numpy()[0], probs.cpu().numpy()[0]))
                if (
                    (final_timestamp := timestamp + pd.Timedelta(seconds=(i+1) * STEP_MINUTES * 60))
                ) and (
                    (final_timestamp := final_timestamp + pd.Timedelta(minutes=90))
                    if timestamp.time() <= datetime.datetime.strptime("4:30", "%H:%M").time() and final_timestamp.time() > datetime.datetime.strptime("4:30", "%H:%M").time()
                    else final_timestamp
                ) and (final_timestamp.time() <= CUTOFF_TIME)
            ]

        json_result = json.dumps([[
            model_name,
            [(ts.strftime('%Y-%m-%d %H:%M:%S'), float(price), float(prob)) for ts, price, prob in predictions]
        ]])

        # print (trade_date, timestamp)

        results.append({
            "trade_date": trade_date,
            "time": timestamp,
            "time_frame": time_frame,
            "result": json_result,
            "last_updated": int(datetime.datetime.utcnow().timestamp()),
            "symbol": "VN30F1M",
            "model_name": model_name
        })

    if results:
        insert_query = '''
            INSERT INTO public.der_model_output (trade_date, "time", time_frame, result, last_updated, symbol, model_name)
            VALUES (:trade_date, :time, :time_frame, :result, :last_updated, :symbol, :model_name)
            ON CONFLICT (trade_date, "time", time_frame, symbol, model_name) DO UPDATE
            SET result = EXCLUDED.result, last_updated = EXCLUDED.last_updated;
        '''
        with engine.connect() as conn:
            conn.execute(text(insert_query), results)
            conn.commit()
        print(f"✅ Saved {len(results)} predictions for {model_name} to database.")
### 
def lstm_predict_and_save(df, model_name, model):
    """ Dự đoán trên dữ liệu mới và lưu kết quả vào database """
    if df.empty:
        print(f"✅ No new data to predict for {model_name}.")
        return
    if df.shape[0] == 0:
        print(f"✅ No new data to predict for {model_name}.")
        return
    df = df[SELECTED_LONG_FEATURES]
    df.fillna(0, inplace=True)
    scaler = joblib.load("scalers/min_max_scaler.pkl")
    df = scaler.transform(df) 
    dataset = SequenceFinancialDataset(df)

    # scaler = joblib.load(model_configs[model_name]["scaler"])

    dataloader = torch.utils.data.DataLoader(df, batch_size=1, shuffle=False)
    results = []
    for idx, (x_short, x_long, _, _, _, _, _) in enumerate(dataloader):
        x_short, x_long = x_short.to(device), x_long.to(device)

         # trade_date = int(df.iloc[idx]['trade_date'])
        trade_date = int(dataset.trade_dates[idx])

        timestamp = pd.to_datetime(dataset.sample_times[idx])
        close = dataset.close_prices[idx]
        time_frame = 15

        with torch.no_grad():
            _, preds, probs = model(x_short, x_long)

            print(f"🔍 Prediction at {timestamp}: {preds.cpu().numpy()[0]} (Probability: {probs.cpu().numpy()[0]})")

            if probs.cpu().numpy()[0][0] < 0.55:
                continue

            CUTOFF_TIME = datetime.datetime.strptime("07:45", "%H:%M").time()

            predictions = [
                (
                    final_timestamp,
                    close * (100 + pred) / 100,
                    prob
                )
                for i, (pred, prob) in enumerate(zip(preds.cpu().numpy()[0], probs.cpu().numpy()[0]))
                if (
                    (final_timestamp := timestamp + pd.Timedelta(seconds=(i+1) * STEP_MINUTES * 60))
                ) and (
                    (final_timestamp := final_timestamp + pd.Timedelta(minutes=90))
                    if timestamp.time() <= datetime.datetime.strptime("4:30", "%H:%M").time() and final_timestamp.time() > datetime.datetime.strptime("4:30", "%H:%M").time()
                    else final_timestamp
                ) and (final_timestamp.time() <= CUTOFF_TIME)
            ]

        json_result = json.dumps([[
            model_name,
            [(ts.strftime('%Y-%m-%d %H:%M:%S'), float(price), float(prob)) for ts, price, prob in predictions]
        ]])

        # print (trade_date, timestamp)

        results.append({
            "trade_date": trade_date,
            "time": timestamp,
            "time_frame": time_frame,
            "result": json_result,
            "last_updated": int(datetime.datetime.utcnow().timestamp()),
            "symbol": "VN30F1M",
            "model_name": model_name
        })

    if results:
        insert_query = '''
            INSERT INTO public.der_model_output (trade_date, "time", time_frame, result, last_updated, symbol, model_name)
            VALUES (:trade_date, :time, :time_frame, :result, :last_updated, :symbol, :model_name)
            ON CONFLICT (trade_date, "time", time_frame, symbol, model_name) DO UPDATE
            SET result = EXCLUDED.result, last_updated = EXCLUDED.last_updated;
        '''
        with engine.connect() as conn:
            conn.execute(text(insert_query), results)
            conn.commit()
        print(f"✅ Saved {len(results)} predictions for {model_name} to database.")


##############################################
#           Run Realtime Prediction Loop
##############################################
def main():
    parser = argparse.ArgumentParser(description="Run model predictions within a date range.")
    parser.add_argument("--start_date", type=int, help="Start date in YYYYMMDD format")
    parser.add_argument("--end_date", type=int, help="End date in YYYYMMDD format")
    args = parser.parse_args()

    while True:
        print("🔍 Checking for new data...")
        for model_name, model in models.items():
            current_date = int(datetime.datetime.now().strftime('%Y%m%d'))
            start_date = args.start_date if args.start_date else current_date
            end_date = args.end_date if args.end_date else current_date

            df_new = get_unpredicted_data(start_date, end_date, model_name)

            # predict_and_save(df_new, model_name, model) #for sklearn model
            lstm_predict_and_save(df_new, model_name, model) # for sequence model (ex: rnn, lstm, gru)

        print("⏳ Waiting for next cycle...")
        time.sleep(60)

if __name__ == "__main__":
    main()

