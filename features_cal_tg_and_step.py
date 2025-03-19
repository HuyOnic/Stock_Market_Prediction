import psycopg2
import pandas as pd
import numpy as np
import logging
from database_utils import get_engine
from sqlalchemy import text
from multiprocessing import Pool, cpu_count
from features_get_encoded_feature_name import get_or_create_feature_name

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
STEP_MINUTES = 15  # Khoảng thời gian của mỗi step (phút)
TARGET_LENGTH = (243 // STEP_MINUTES) + 1  # target_0 đến target_15
MAX_STEPS = (243 // STEP_MINUTES) + 1      # step_0 đến step_15
START_HOUR = 2  # Giả sử dữ liệu bắt đầu từ 2h mỗi phiên
DELTA = 0.1   # Khoảng chia target thành các nhãn

# Hàm chuyển đổi phần trăm thay đổi thành nhãn, ép kiểu về int của Python
def percent_to_label(percent, delta=DELTA):
    percent = np.clip(percent, -10.0, 10.0) # Convert -10 and 10 to float
    return int(round(percent / delta))

def label_to_percent(label, delta=DELTA):
    return label * delta

# Hàm tạo các cột nếu chưa tồn tại trong bảng
def create_missing_columns():
    engine = get_engine()
    with engine.connect() as conn:
        alter_commands = []
        # Các cột step là kiểu số thực (float)
        for j in range(MAX_STEPS + 1):
            feature_name = get_or_create_feature_name(f'step_{j}')
            alter_commands.append(f'ADD COLUMN IF NOT EXISTS "{feature_name}" float')  # Add j to colname
        # Các cột target là kiểu số nguyên (integer)
        for j in range(TARGET_LENGTH):
            alter_commands.append(f'ADD COLUMN IF NOT EXISTS "target_{j}" integer')  # Add j to colname
        alter_query = 'ALTER TABLE public."der_1m_feature" ' + ", ".join(alter_commands) + ";"
        logger.info("✅ Tạo cột nếu chưa tồn tại...")
        conn.execute(text(alter_query))
        conn.commit()


def compute_future_values(group):
    group = group.sort_values('time')  # Đảm bảo dữ liệu theo thứ tự thời gian

    # Tạo cột 'time_prev' với giá trị 15 phút trước đó
    group['time_prev'] = group['time'] - pd.Timedelta(minutes=STEP_MINUTES)

    # Xác định khoảng nghỉ trưa (giả sử nghỉ từ 04:30:00 đến 06:00:00)
    mask_lunch_break = (group['time_prev'].dt.time >= pd.to_datetime("04:30:00").time()) & \
                       (group['time_prev'].dt.time < pd.to_datetime("06:00:00").time())

    # 🔥 Fix: Đối với dữ liệu đầu tiên sau nghỉ trưa (>= 06:00:00)
    # Tìm giá trị gần nhất trước đó (trong phiên sáng) để tham chiếu
    first_after_lunch = group.loc[group['time'].dt.time >= pd.to_datetime("06:00:00").time()].index.min()
    last_before_lunch = group.loc[group['time'].dt.time < pd.to_datetime("04:30:00").time()].index.max()

    if pd.notna(first_after_lunch) and pd.notna(last_before_lunch):
        group.at[first_after_lunch, 'time_prev'] = group.at[last_before_lunch, 'time']

    # Loại bỏ NaN trong time_prev trước khi merge_asof
    group = group.dropna(subset=['time_prev'])

    # **🔥 Fix: Sắp xếp lại `group` theo `time_prev` trước khi merge**
    group = group.sort_values('time_prev')

    # Nếu nhóm rỗng sau khi loại bỏ, trả về toàn 0
    if group.empty:
        return np.zeros(len(group))

    # Dùng merge_asof để tìm giá trị close của 15 phút trước
    group = pd.merge_asof(
        group,
        group[['time', 'close']].rename(columns={'time': 'time_prev', 'close': 'close_prev'}),
        on='time_prev',
        direction='backward'
    )

    # Tính phần trăm thay đổi
    group['future_change'] = ((group['close'] - group['close_prev']) / group['close_prev']) * 100

    # Điền 0 nếu không tìm thấy giá trị hợp lệ
    group['future_change'] = group['future_change'].fillna(0)

    return group['future_change'].to_numpy()


def compute_labels_for_group(group_values, batch, date):
    labels = []
    date_batch = batch[batch['trade_date'] == date]
    for group_idx in date_batch['group_index']:
        start_idx = group_idx + 1
        end_idx = start_idx + TARGET_LENGTH
        future_changes = [percent_to_label(p) for p in group_values[start_idx:end_idx]]
        padded_changes = np.pad(
            future_changes,
            (0, max(0, TARGET_LENGTH - len(future_changes))),
            mode='constant'
        )
        labels.append(padded_changes)
    return np.vstack(labels) if labels else np.array([])


def process_batch(batch):
    engine = get_engine()
    with engine.connect() as conn:
        batch = batch.reset_index(drop=True)
        batch = batch[batch['timestamp'] % STEP_MINUTES == 0].reset_index(drop=True)

        # Tạo index theo bước 15 phút

        # batch['group_index'] = (batch['timestamp'] // STEP_MINUTES).astype(int)
        batch['group_index'] = batch.groupby('trade_date').cumcount()

        batch['step_index'] = np.clip(batch['group_index'], 0, MAX_STEPS)
        step_one_hot_matrix = np.eye(MAX_STEPS + 1)[batch['step_index']]

        # Áp dụng groupby theo 'date'
        future_values_series = batch.groupby('trade_date', group_keys=False).apply(compute_future_values)

        future_values = {date: future_values_series[date] for date in batch['trade_date'].unique()}  # Removed .to_numpy()

        # print (batch)
        # print (future_values_series)

        # Tạo nhãn chính xác cho từng dòng trong batch
        future_labels = {}
        for date in batch['trade_date'].unique():
            group_values = future_values.get(date)
            if group_values is not None:
                labels = compute_labels_for_group(group_values, batch, date)
                future_labels[date] = labels
            else:
                future_labels[date] = np.array([])

        # print (future_labels)

        batch_update_data = []
        for i in range(len(batch)):
            result_row = {'time': batch['time'].iloc[i].to_pydatetime()}
            for j in range(MAX_STEPS + 1):
                feature_name = get_or_create_feature_name(f'step_{j}')
                result_row[feature_name] = float(step_one_hot_matrix[i, j])

            group_step_index = batch['group_index'].iloc[i]
            current_date = batch['trade_date'].iloc[i]

            matrix = future_labels.get(current_date)
            if matrix is not None and matrix.size > 0 and 0 <= group_step_index < matrix.shape[0]:
                for j in range(TARGET_LENGTH):
                    result_row[f'target_{j}'] = int(matrix[group_step_index, j])
            else:
                for j in range(TARGET_LENGTH):
                    result_row[f'target_{j}'] = 0

            batch_update_data.append(result_row)

        if batch_update_data:
            # Chuẩn bị danh sách các cột
            step_columns = [f'"{get_or_create_feature_name(f"step_{j}")}"' for j in range(MAX_STEPS + 1)]
            target_columns = [f'"target_{i}"' for i in range(TARGET_LENGTH)]
            all_columns = ['"time"'] + step_columns + target_columns

            # Tạo danh sách giá trị
            values_str = ", ".join([
                f"('{row['time']}'::timestamp, " +
                ", ".join([str(row[feature]) for feature in step_columns]) + ", " +
                ", ".join([str(row[f'target_{i}']) for i in range(TARGET_LENGTH)]) +
                ")"
                for row in batch_update_data
            ])

            # Tạo truy vấn cập nhật hàng loạt
            update_query = f"""
            INSERT INTO public."der_1m_feature" ({", ".join(all_columns)})
            VALUES {values_str}
            ON CONFLICT ("time") DO UPDATE SET
            {", ".join([f"{col} = EXCLUDED.{col}" for col in step_columns + target_columns])};
            """

            # Thực thi truy vấn cập nhật hàng loạt
            conn.execute(text(update_query))
            conn.commit()



if __name__ == "__main__":
    try:
        engine = get_engine()
        with engine.connect() as conn:
            logger.info("✅ Kết nối PostgreSQL thành công!")

            # Tạo các cột cần thiết nếu chưa tồn tại
            create_missing_columns()

            # --- XÓA HẾT CÁC GIÁ TRỊ HIỆN TẠI TRONG CÁC CỘT computed (step, target)
            # cho những dòng mà phút của "time" không phải là bội số của 15
            clear_query = text(
                """
                UPDATE public."der_1m_feature"
                SET """ + ", ".join([f'"{get_or_create_feature_name(f"step_{j}")}" = NULL' for j in range(MAX_STEPS + 1)] + [f'"target_{i}" = NULL' for i in range(TARGET_LENGTH)]) + """
                WHERE EXTRACT(MINUTE FROM "time")::int % 15 <> 0;
                """
            )
        logger.info("✅ Đang xóa các giá trị computed ở những dòng không có phút chẵn 15...")
        conn.execute(clear_query)
        conn.commit()
        # --- KẾT THÚC PHẦN XÓA

        # Đọc dữ liệu cần thiết (bao gồm các dòng có phút chẵn và không chẵn)
        query = (
            'SELECT f1."time", f1."trade_date", ohlc."close" '
            'FROM public."der_1m_feature" f1 '
            'JOIN public."der_1m" ohlc ON f1."time" = ohlc."time" '
            'ORDER BY f1."time";'
        )
        df = pd.read_sql(query, conn)
        logger.info("✅ Đọc dữ liệu thành công!")

        df['time'] = pd.to_datetime(df['time'])
        # Tính timestamp theo số phút thật từ giờ bắt đầu, không làm tròn về bội số của 15
        df['timestamp'] = ((df['time'].dt.hour - START_HOUR) * 60 + df['time'].dt.minute)

        # Tạo batch theo ngày để xử lý song song
        unique_dates = df['trade_date'].unique()
        # unique_dates = df['date'].unique()[:1]

        batch_size = max(1, len(unique_dates) // cpu_count())
        # batch_size = 1

        batches = [df[df['trade_date'].isin(batch)].copy() for batch in np.array_split(unique_dates, batch_size)]



        with Pool(cpu_count()) as pool:
            pool.map(process_batch, batches)

        # for batch in batches:
        #     process_batch(batch)

        logger.info("✅ Dữ liệu đã được cập nhật thành công!")

    except Exception as e:
        logger.error(f"❌ Lỗi: {str(e)}")
        raise (e)

    finally:
        logger.info("Đã hoàn tất xử lý.")