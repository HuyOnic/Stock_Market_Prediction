import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from datetime import timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys, os 
sys.path.append(os.getcwd())
from database_utils import get_engine
from sqlalchemy import text
import joblib

from sklearn.decomposition import PCA
import joblib

##############################################
# Settings
##############################################

# Thời gian start và end (thay đổi nếu cần)
START_DATE = "20200102"  # Định dạng YYYY-MM-DD hoặc None nếu không giới hạn
# START_DATE = "20240627"  # Định dạng YYYY-MM-DD hoặc None nếu không giới hạn

END_DATE = "20250228"    # Định dạng YYYY-MM-DD hoặc None nếu không giới hạn

# Constants
STEP_MINUTES = 15  # Khoảng thời gian của mỗi step (phút)
TARGET_LENGTH = (243 // STEP_MINUTES) + 1  # target_0 đến target_15
MAX_STEPS = (243 // STEP_MINUTES) + 1      # step_0 đến step_15
START_HOUR = 2  # Giả sử dữ liệu bắt đầu từ 2h mỗi phiên
DELTA = 0.1   # Khoảng chia target thành các nhãn

SCALE_FACTOR = 1000

# ✅ Các đặc trưng thời gian (day_0 đến day_6 và hour_2 đến hour_7)
DAY_FEATURES = [f'f{i}' for i in range(6, 13)]
HOUR_FEATURES = [f'f{i}' for i in range(13, 19)]
MONTH_FEATURES = [f'f{i}' for i in range(19, 31)]
LEAP_YEAR_FEATURES = [f'f{i}' for i in range(31, 35)]  # 0 ~ 3
QUARTER_FEATURES = [f'f{i}' for i in range(35, 39)]
HOLIDAYS_FEATURES = ["f176", "f177", "f178", "f179", "f180"]

# ✅ Tập hợp tất cả các time features
TIME_FEATURES = DAY_FEATURES + HOUR_FEATURES + MONTH_FEATURES + LEAP_YEAR_FEATURES + QUARTER_FEATURES + HOLIDAYS_FEATURES

TA_FEATURES = [f'f{i}' for i in range(49, 176)]

ORDERBOOK_INTRADAY_TIMESTEP_FEATURES = [f'f{i}' for i in range(181, 240)] + [f'f{i}' for i in range(319, 349)]

EOD_FEATURES = [f'f{i}' for i in range(240, 319)]

INDEX_FEATURES = []
INDEX_FEATURES = (
        [f'f{i}' for i in range(349, 353)] +
        [f'f{i}' for i in range(353, 361)] +
        [f'f{i}' for i in range(361, 369)] +
        [f'f{i}' for i in range(369, 376)] +
        [f'f{i}' for i in range(376, 396)] +
        [f'f{i}' for i in range(400, 628)] +
        # ["f628"] +
        [f'f{i}' for i in range(629, 753)] +
        ["f757"] 
        # [f'f{i}' for i in range(828, 842)] +
        # [f'f{i}' for i in range(849, 870)] +
        # [f'f{i}' for i in range(877, 891)] +
        # [f'f{i}' for i in range(1018, 1046)] +
        # ['f945'] +
        # [f'f{i}' for i in range(1053, 1258)]
)

SHORT_FEATURES = ['f39', 'f40', 'f41',
                                        'f42', 'f43'] #+ TA_FEATURES
LONG_FEATURES = (['f45', 'f48', 'f47'] +
                 ['f39', 'f40', 'f41'] +
                 ['f42', 'f43'] +
                 TIME_FEATURES
                + TA_FEATURES
                 # + ORDERBOOK_INTRADAY_TIMESTEP_FEATURES
                 # + INDEX_FEATURES

                #  + ["f1102", "f1101", "f1100", "f1099", "f1098", "f1097", "f1096", "f1095", "f1094", "f1093", "f1092", "f1091", "f1090", "f1089", "f1076", "f1075", "f1074", "f1073", "f1072", "f1071", "f1070", "f1066", "f1065", "f1064", "f1063", "f1062", "f1061", "f1060", "f1059", "f1058", "f1057", "f1056", "f1055", "f1054", "f1053", "f945"] #

                #  + ["f1017", "f1016", "f1015", "f1014", "f1013", "f1012", "f1011", "f1010", "f1009", "f419", "f418", "f417", "f416", "f415", "f414", "f413", "f412", "f411", "f410", "f409", "f408", "f407", "f406", "f405", "f404", "f403", "f402", "f401", "f400", "f395", "f394", "f393", "f392", "f391", "f390", "f389", "f388", "f387", "f386", "f385", "f384", "f383", "f382", "f381", "f380", "f379", "f378", "f377", "f376", "f323", "f322", "f321", "f320", "f319", "f185", "f184", "f183", "f182", "f181"] #

                #  + ["f1045", "f1044", "f1043", "f1042", "f1041", "f1040", "f1039", "f1038", "f1037", "f1036", "f1035", "f1034", "f1033", "f1032", "f1031", "f1030", "f1029", "f1028", "f1027", "f1026", "f1025", "f1024", "f1023", "f1022", "f1021", "f1020", "f1019", "f1018", "f890", "f889", "f888", "f887", "f886", "f885", "f884", "f883", "f882", "f881", "f880", "f879", "f878", "f877", "f876", "f875", "f874", "f873", "f872", "f871", "f870", "f869", "f868", "f867", "f866", "f865", "f864", "f863", "f862", "f861", "f860", "f859", "f858", "f857", "f856", "f855", "f854", "f853", "f852", "f851", "f850", "f849", "f848", "f847", "f846", "f845", "f844", "f843", "f842", "f841", "f840", "f839", "f838", "f837", "f836", "f835", "f834", "f833", "f832", "f831", "f830", "f829", "f828", "f827", "f826", "f825", "f824", "f823", "f822", "f821", "f820", "f819", "f818", "f817", "f816", "f815", "f814", "f813", "f812", "f811", "f810", "f809", "f808", "f807", "f806", "f805", "f804", "f803", "f802", "f801", "f800", "f799", "f798", "f797", "f796", "f795", "f794", "f793", "f792", "f791", "f790", "f789", "f788", "f787", "f786", "f785", "f784", "f783", "f782", "f781", "f780", "f779", "f778", "f777", "f776", "f775", "f774", "f773", "f772", "f771", "f770", "f769", "f768", "f767", "f766", "f765", "f764", "f763", "f762", "f761", "f760", "f759", "f758", "f459", "f458", "f457", "f456", "f455", "f454", "f453", "f452", "f451", "f450", "f449", "f448", "f447", "f446", "f445", "f444", "f443", "f442", "f441", "f440", "f439", "f438", "f437", "f436", "f435", "f434", "f433", "f432", "f431", "f430", "f429", "f428", "f427", "f426", "f425", "f424", "f423", "f422", "f421", "f420", "f195", "f194", "f193", "f192", "f191"] #

                #  # + ["f744", "f745", "f753", "f1077", "f1078", "f1079", "f1080", "f1081", "f1082", "f1083", "f1084", "f1085", "f1086", "f1087", "f1088"] #

                #  + ["f935", "f937", "f938", "f939", "f940", "f941", "f946", "f947", "f948", "f949", "f950", "f951", "f952", "f953", "f954", "f955", "f956", "f957", "f1108", "f1109", "f1110", "f1111", "f1112", "f1113", "f1114", "f1115", "f1116", "f1117", "f1118", "f1119", "f1120", "f1121", "f1122", "f1123", "f1124", "f1125", "f1126", "f1127", "f1128", "f1129", "f1130"] #

                #  # + ["f186", "f187", "f188", "f189", "f190", "f396", "f397", "f398", "f399", "f460", "f461", "f462", "f463", "f464", "f465", "f466", "f467", "f468", "f469", "f470", "f471", "f472", "f473", "f474", "f475", "f476", "f477", "f478", "f479", "f480", "f481", "f482", "f483", "f484", "f485", "f486", "f487", "f488", "f489", "f490", "f491", "f492", "f493", "f494", "f495", "f496", "f497", "f498", "f499"] # 
                #  #
                #  # +

                 )

ALL_NEEDED_FEATURES = ["time", "trade_date"] + list(set(SHORT_FEATURES + LONG_FEATURES)) + ["target_0", "target_1", "target_2", "target_3", "target_4", "target_5", "target_6", "target_7", "target_8", "target_9", "target_10", "target_11", "target_12", "target_13", "target_14", "target_15", "target_16"] + [f"target_close_{i}" for i in range(17)] + [f"target_pct_{i}" for i in range(17)]

# LONG_FEATURES += EOD_FEATURES

# print (len(TIME_FEATURES), len(TA_FEATURES))
# exit()


def load_data(load_from_db=False,
              table_name="der_1m_feature",
              eod_table="der_1d_feature",
              der_1m_table="der_1m",
              csv_path="../../data/all_data.csv", save_to_file=True, start_date=START_DATE, end_date=END_DATE):
    """
    Load dữ liệu từ database hoặc từ CSV. Sau đó merge với dữ liệu EOD (chỉ số kết phiên hôm trước).
    """
    data_source = csv_path if not load_from_db else "Database"
    print(f"Loading data from {data_source}")
    if load_from_db:
        # Lấy engine từ database_utils
        engine = get_engine()

        # Xây dựng danh sách cột cần SELECT cho dữ liệu intraday
        select_columns = ", ".join([f'"{col}"' for col in ALL_NEEDED_FEATURES])

        # Truy vấn dữ liệu intraday
        query_str = f'''
            SELECT {select_columns} 
            FROM public."{table_name}"
            WHERE 1=1
            AND "symbol" = 'VN30F1M'
        '''
        if start_date:
            query_str += f' AND "trade_date" >= \'{start_date}\''
        if end_date:
            query_str += f' AND "trade_date" <= \'{end_date}\''

        query_str += ' ORDER BY "time"'
        query = text(query_str)
        df = pd.read_sql(query, con=engine)

        # ✅ Truy vấn dữ liệu EOD (chỉ lấy ngày và các chỉ số)
        eod_query = f'''
            SELECT "trade_date" AS eod_date, * 
            FROM public."{eod_table}"
            WHERE "trade_date" >= '{start_date}' AND "trade_date" <= '{end_date}'
            ORDER BY "trade_date"
        '''
        eod_df = pd.read_sql(text(eod_query), con=engine)

        # ✅ Dịch dữ liệu EOD lùi 1 ngày (vì chỉ có thể biết EOD của ngày hôm trước)
        # eod_df["eod_date"] = pd.to_datetime(eod_df["eod_date"])
        # eod_df["merge_date"] = eod_df["eod_date"] + pd.Timedelta(days=1)  # Dịch ngày lên 1 ngày

        eod_df["merge_date"] = eod_df["eod_date"]  # Dịch ngày lên 1 ngày
        eod_df.drop(columns=["eod_date"], inplace=True)

        # ✅ Merge với dữ liệu intraday theo "date"
        # df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.merge(eod_df, left_on="trade_date", right_on="merge_date", how="left")
        df.drop(columns=["merge_date"], inplace=True)
        df["trade_date"] = df["trade_date_x"]

        # ✅ Truy vấn dữ liệu close_price từ der_1m_table
        close_price_query = f'''
            SELECT "time", "symbol", "close" AS close
            FROM public."{der_1m_table}"
            WHERE "trade_date" >= '{start_date}' AND "trade_date" <= '{end_date}'
            AND "symbol" = 'VN30F1M'
        '''
        close_price_df = pd.read_sql(text(close_price_query), con=engine)

        # ✅ Merge với close_price_df theo "trade_date" và "time"
        df = df.merge(close_price_df, on=["time"], how="left")

        # ✅ Xử lý các giá trị thiếu (nếu cần)
        df["close"].fillna(method="ffill", inplace=True)  # Điền giá trị NaN bằng giá trước đó nếu có


        # Lưu ra CSV nếu cần
        if save_to_file:
            df.to_csv(csv_path, index=False)
            print(f"✅ Dữ liệu đã load từ DB, merge với EOD và lưu ra CSV: {csv_path}")

    else:
        # Đọc từ CSV
        df = pd.read_csv(csv_path)
        print(f"✅ Dữ liệu đã load từ CSV: {csv_path}")

    return df


##############################################
# Dataset Class
##############################################
class FinancialDataset(Dataset):
    def __init__(self, data, window_size=60, step_minutes=STEP_MINUTES):
        """
        - Chỉ lấy dữ liệu tại các mốc thời gian thỏa mãn `STEP_MINUTES`.
        - Short-term features (chuỗi dữ liệu trước thời điểm dự báo).
        - Long-term features lấy tại thời điểm hiện tại.
        - Target, mask, percentages, labels lấy trực tiếp từ dataset.
        """
        self.window_size = window_size
        self.X_short_term = []
        self.X_long_term = []
        self.y = []
        self.masks = []
        self.percentages = []
        self.labels = []
        self.sample_dates = []
        self.sample_times = []
        self.close_prices = []
        self.trade_dates = []

        self.feature_names = LONG_FEATURES
        # self.feature_names = ['f45', 'f104', 'f105', 'f245', 'f1125', 'f368', 'f237', 'f531', 'f1171', 'f1160', 'f42', 'f366', 'f1101', 'f481', 'f1100', 'f355', 'f88', 'f490', 'f239', 'f354', 'f757', 'f47', 'f271', 'f48', 'f1099', 'f303', 'f438', 'f453', 'f250', 'f359', 'f360', 'f1074', 'f1098', 'f487', 'f240', 'f411', 'f582', 'f1126', 'f1155', 'f83', 'f572', 'f253', 'f521', 'f530', 'f341', 'f367', 'f80', 'f1157', 'f68', 'f1073']

        # Chuyển toàn bộ data thành numpy để truy xuất nhanh
        short_term_features = data[SHORT_FEATURES].values
        long_term_features = data[self.feature_names].values
        y_data = data.filter(regex=r"^target_close_").values
        mask_data = data.filter(regex=r"^target_\d+$").values
        percentage_data = data.filter(regex=r"^target_pct_\d+$").values
        label_data = data.filter(regex=r"^target_\d+$").values
        time_data = pd.to_datetime(data["time"])  # Chuyển thành datetime để xử lý thời gian

        # ✅ Chỉ giữ các mốc thời gian cách nhau STEP_MINUTES từ đầu phiên
        start_of_session = time_data.dt.normalize() + pd.Timedelta(hours=START_HOUR)

        # valid_indices = (time_data - start_of_session).dt.total_seconds() % (step_minutes * 60) == 0
        valid_indices = np.ones(len(time_data), dtype=bool)  # Giữ tất cả các mốc thời gian

        filtered_indices = np.where(valid_indices)[0]  # Lấy index của các điểm cần dự báo

        for i in filtered_indices:
            # ✅ Lấy cửa sổ dữ liệu `window_size` bước về trước cho short-term features
            start_idx = max(0, i - window_size + 1)
            short_term_window = short_term_features[start_idx:i + 1]

            # Nếu thiếu dữ liệu ở đầu, pad bằng 0
            if len(short_term_window) < window_size:
                pad_length = window_size - len(short_term_window)
                short_term_window = np.pad(short_term_window, ((pad_length, 0), (0, 0)), mode='constant')

            # Lưu lại các giá trị vào danh sách
            self.X_short_term.append(short_term_window)
            self.X_long_term.append(long_term_features[i])
            self.y.append(y_data[i])
            self.masks.append(mask_data[i])
            self.percentages.append(percentage_data[i])

            # self.labels.append(label_data[i])
            self.labels.append(np.where(label_data[i] < 0, -1, np.where(label_data[i] > 0, 1, 0)))

            self.sample_dates.append(pd.to_datetime(data["trade_date_x"].iloc[i], format='%Y%m%d', errors='coerce'))
            self.trade_dates.append(data["trade_date_x"].iloc[i])
            self.sample_times.append(data["time"].iloc[i])
            self.close_prices.append(data["close"].iloc[i])


        # Chuyển thành numpy arrays để tối ưu tốc độ truy xuất
        self.X_short_term = np.array(self.X_short_term)
        self.X_long_term = np.array(self.X_long_term)
        self.y = np.array(self.y)
        self.masks = np.array(self.masks)
        self.percentages = np.array(self.percentages)
        self.labels = np.array(self.labels)
        self.close_prices = np.array(self.close_prices)

    def scale_data(self, scaler):
        # Scale short-term price features
        num_price_features = 5
        price_data = self.X_short_term[:, :, :num_price_features]
        price_data_scaled = scaler['short_term'].transform(price_data.reshape(-1, num_price_features)).reshape(price_data.shape)

        self.X_short_term = np.concatenate([price_data_scaled, self.X_short_term[:, :, num_price_features:]], axis=2)

        # Scale long-term price features
        num_long_term_price_features = 3
        long_term_price_data = self.X_long_term[:, :num_long_term_price_features]
        long_term_price_data_scaled = scaler['long_term'].transform(long_term_price_data)

        self.X_long_term = np.concatenate([long_term_price_data_scaled, self.X_long_term[:, num_long_term_price_features:]], axis=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.tensor(self.X_short_term[idx], dtype=torch.float32),
                torch.tensor(self.X_long_term[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32),
                torch.tensor(self.masks[idx], dtype=torch.float32),
                torch.tensor(self.percentages[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long),
                torch.tensor(self.close_prices[idx], dtype=torch.long)
                )

class SequenceFinancialDataset(Dataset):
    def __init__(self, data: pd.DataFrame, selected_long_features=LONG_FEATURES, window_size: int = 60, seq_length: int = 1000, step_minutes=STEP_MINUTES):
        self.data = data
        self.window_size = window_size
        self.seq_length = seq_length
        self.selected_long_features = selected_long_features
        self.prepare_sequence_data()

    def prepare_sequence_data(self):
        time_data = pd.to_datetime(self.data["time"])
        # ✅ Chỉ giữ các mốc thời gian cách nhau STEP_MINUTES từ đầu phiên
        start_of_session = time_data.dt.normalize() + pd.Timedelta(hours=START_HOUR)
        # valid_indices = (time_data - start_of_session).dt.total_seconds() % (step_minutes * 60) == 0
        valid_indices = np.ones(len(time_data), dtype=bool)  # Giữ tất cả các mốc thời gian
        filtered_indices = np.where(valid_indices)[0]  # Lấy index của các điểm cần dự báo

        self.X_short_term = []
        short_term_features = self.data[SHORT_FEATURES].values
        for i in filtered_indices:
            # ✅ Lấy cửa sổ dữ liệu `window_size` bước về trước cho short-term features
            start_idx = max(0, i - self.window_size + 1)
            short_term_window = short_term_features[start_idx:i + 1]
            # Nếu thiếu dữ liệu ở đầu, pad bằng 0
            if len(short_term_window) < self.window_size:
                pad_length = self.window_size - len(short_term_window)
                short_term_window = np.pad(short_term_window, ((pad_length, 0), (0, 0)), mode='constant')
            # Lưu lại các giá trị vào danh sách
            self.X_short_term.append(short_term_window)

        self.X_long_term = self.data[self.selected_long_features].iloc[valid_indices]
        self.labels = self.data.filter(regex=r"^target_\d+$").map(self.standard_label)
        self.y = self.data.filter(regex=r"^target_close_\d+$").iloc[valid_indices]
        self.masks = self.data.filter(regex=r"^target_\d+$").iloc[valid_indices]
        self.percentages = self.data.filter(regex=r"^target_pct_\d+$").iloc[valid_indices]
        self.trade_dates = self.data["trade_date_x"].iloc[valid_indices]
        self.sample_times = self.data["time"].iloc[valid_indices]
        self.close_prices = self.data["close"].iloc[valid_indices]
    def standard_label(self, label):
        if label>1:
            return 1
        elif label<-1:
            return -1
        return label

    def __len__(self):
        return len(self.data)-self.seq_length
    
    def __getitem__(self, index):
        return torch.tensor(np.array(self.X_short_term[index:index+self.seq_length]), dtype=torch.float32), \
                torch.tensor(self.X_long_term.iloc[index:index+self.seq_length].values, dtype=torch.float32), \
                torch.tensor(self.y.iloc[index:index+self.seq_length].values, dtype=torch.float32), \
                torch.tensor(self.masks.iloc[index:index+self.seq_length].values, dtype=torch.float32), \
                torch.tensor(self.percentages.iloc[index:index+self.seq_length].values, dtype=torch.float32), \
                torch.tensor(self.labels[index:index+self.seq_length].values, dtype=torch.long), \
                torch.tensor(self.close_prices.iloc[index:index+self.seq_length].values, dtype=torch.long)

##############################################
# Scaling
##############################################
def compute_scalers(train_dataset):
    short_term_scaler = StandardScaler()
    long_term_scaler = StandardScaler()

    num_price_features = 5
    short_term_data = train_dataset.dataset.X_short_term[:, :, :num_price_features].reshape(-1, num_price_features)
    long_term_data = train_dataset.dataset.X_long_term[:, :3]

    short_term_scaler.fit(short_term_data)
    long_term_scaler.fit(long_term_data)

    return {'short_term': short_term_scaler, 'long_term': long_term_scaler}

def fit_min_max_scalers(train_df: pd.DataFrame, scaler_path: str='min_max_scaler.pkl'):
    long_term_scaler = MinMaxScaler()
    long_term_scaler.fit(train_df[LONG_FEATURES])
    folder_path = 'scalers'
    os.makedirs(folder_path, exist_ok=True)
    joblib.dump(long_term_scaler, os.path.join(folder_path, scaler_path))
    return long_term_scaler

def transform_min_max_scalers(df: pd.DataFrame, scaler_path: str='scaler/min_max_scaler.pkl'):
    scaler = joblib.load(scaler_path)
    return pd.DataFrame(scaler.transform(df[LONG_FEATURES]), index=df.index, columns=df.columns)

##############################################
# Load Train/Val/Test
##############################################
def get_train_val_test(num_train_day: int=120, num_val_day: int=30, num_test_day: int=30):
    # data = pd.read_csv('D:\Quant_Prj_Goline\processed_data\OHLC_VN30F1M_processed.csv')
    # data = pd.read_csv('../../data/all_data.csv')
    # data = pd.read_csv('../../data/all_data.csv', nrows=1000)

    data = load_data(load_from_db=False, csv_path="data/all_data.csv")
    # data = load_data(load_from_db=False)
    data.fillna(0, inplace=True)
    zero_cols = data[LONG_FEATURES].columns[data[LONG_FEATURES].sum(axis=0)==0].tolist()
    selected_long_features = list(filter(lambda nonzero: nonzero not in zero_cols, LONG_FEATURES))

    if 'time' in data.columns:
        data.index = pd.to_datetime(data['time'])
    last_date = data.index[-1]
    test_start_date = last_date - timedelta(days=num_test_day)
    val_start_date = test_start_date - timedelta(days=num_val_day)
    train_start_date = val_start_date - timedelta(days=num_train_day)

    train_df = data.loc[train_start_date: val_start_date]
    val_df = data.loc[val_start_date: test_start_date]  
    test_df = data.loc[test_start_date:]
    #data_preprocessing
    scaler = fit_min_max_scalers(train_df)
    train_df.loc[:,LONG_FEATURES] = pd.DataFrame(scaler.transform(train_df.loc[:,LONG_FEATURES]), index=train_df.index, columns=train_df[LONG_FEATURES].columns)
    val_df.loc[:,LONG_FEATURES] = pd.DataFrame(scaler.transform(val_df.loc[:,LONG_FEATURES]), index=val_df.index, columns=val_df[LONG_FEATURES].columns)
    test_df.loc[:,LONG_FEATURES] = pd.DataFrame(scaler.transform(test_df.loc[:,LONG_FEATURES]), index=test_df.index, columns=test_df[LONG_FEATURES].columns)

    train_dataset = SequenceFinancialDataset(train_df, selected_long_features=selected_long_features)
    val_dataset = SequenceFinancialDataset(val_df, selected_long_features=selected_long_features)
    test_dataset = SequenceFinancialDataset(test_df, selected_long_features=selected_long_features)

    print(f"Train: From {train_start_date} to {val_start_date} ({num_train_day} days)")
    print(f"Validation: {val_start_date} to {test_start_date} ({num_val_day} days)") 
    print(f"Test: {test_start_date} to {last_date} ({num_test_day} days)")

    return data, train_dataset, val_dataset, test_dataset, '', '', ''


def apply_pca_auto(train_dataset, val_dataset, test_dataset, variance_threshold=0.95, save_path="pca.pkl"):
    """
    Tự động chọn số chiều PCA để giữ ít nhất `variance_threshold` phương sai của dữ liệu train.

    Parameters:
        - train_dataset (Subset): Dữ liệu training.
        - val_dataset (Subset): Dữ liệu validation.
        - test_dataset (Subset): Dữ liệu test.
        - variance_threshold (float): Tỷ lệ phương sai muốn giữ lại (0 < variance_threshold <= 1).
        - save_path (str): Đường dẫn lưu PCA model đã được fit.

    Returns:
        - PCA-fitted datasets (train, val, test) dưới dạng numpy arrays.
        - Số lượng thành phần chính đã chọn.
    """

    # Lấy dữ liệu từ Subset Dataset
    X_train = np.array([train_dataset.dataset.X_long_term[i] for i in train_dataset.indices])
    X_val = np.array([val_dataset.dataset.X_long_term[i] for i in val_dataset.indices])
    X_test = np.array([test_dataset.dataset.X_long_term[i] for i in test_dataset.indices])

    # ✅ Fit PCA với số chiều không giới hạn, để kiểm tra bao nhiêu cần giữ lại `variance_threshold`
    pca_full = PCA().fit(X_train)

    # ✅ Tính toán số chiều cần thiết để giữ lại `variance_threshold` phương sai
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    print(f"🔹 PCA chọn {n_components} thành phần chính để giữ lại {variance_threshold * 100:.1f}% phương sai")

    # ✅ Fit lại PCA với số chiều đã chọn
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)

    # ✅ Áp dụng PCA đã fit lên tập val & test
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    # ✅ Lưu PCA đã được fit để dùng lại
    joblib.dump(pca, save_path)
    print(f"✅ PCA model saved to {save_path}")

    return X_train_pca, X_val_pca, X_test_pca, n_components


if __name__ == "__main__":
    _, train_dataset, val_dataset, test_dataset, _, _, _ = get_train_val_test()
    print("✅ Data loaded successfully!")

    # ✅ Chạy PCA tự động chọn số chiều để giữ ít nhất 95% phương sai
    X_train_pca, X_val_pca, X_test_pca, num_components = apply_pca_auto(train_dataset, val_dataset, test_dataset, variance_threshold=0.95)

    print(f"Số chiều được chọn: {num_components}")

