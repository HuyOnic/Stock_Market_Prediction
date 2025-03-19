from sqlalchemy import text
from database_utils import get_engine_prod  # Import từ database_utils.py

def get_or_create_feature_name(origin_feature_name):
    engine = get_engine_prod()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT feature_name FROM public.convert_features_name
            WHERE origin_feature_name = :origin_feature_name
        """), {"origin_feature_name": origin_feature_name})
        row = result.fetchone()

        if row:
            return row[0]

        # Nếu không có, tạo feature_name mới
        result = conn.execute(text("""
            SELECT COALESCE(MAX(feature_index), 0) + 1 FROM public.convert_features_name
        """))
        new_index = result.scalar()
        new_feature_name = f"f{new_index}"

        conn.execute(text("""
            INSERT INTO public.convert_features_name (origin_feature_name, feature_name, feature_index)
            VALUES (:origin_feature_name, :feature_name, :feature_index)
        """), {"origin_feature_name": origin_feature_name, "feature_name": new_feature_name, "feature_index": new_index})
        conn.commit()
        return new_feature_name

def get_feature_name(origin_feature_name):
    """
    Truy vấn feature_name đã mã hóa từ bảng convert_features_name.
    Nếu không có, trả về None.
    """
    engine = get_engine_prod()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT feature_name FROM public.convert_features_name
            WHERE origin_feature_name = :origin_feature_name
        """), {"origin_feature_name": origin_feature_name})
        row = result.fetchone()

        return row[0] if row else None

def get_origin_feature_name(feature_name):
    """
    Trả về origin_feature_name từ feature_name trong bảng convert_features_name.
    Nếu feature_name không tồn tại, trả về None.
    """
    engine = get_engine_prod()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT origin_feature_name FROM public.convert_features_name
            WHERE feature_name = :feature_name
        """), {"feature_name": feature_name})
        row = result.fetchone()

        return row[0] if row else None  # Trả về origin_feature_name nếu tồn tại, nếu không thì None

if __name__ == "__main__":
    features_list = ["day_of_week", "hour_of_day", "month", "years_since_last_leap", 
                     "quarter", "day_0", "day_1", "day_2", "day_3", "day_4", "day_5", "day_6", "hour_2", 
                     "hour_3", "hour_4", "hour_5", "hour_6", "hour_7", "month_1", "month_2", "month_3", 
                     "month_4", "month_5", "month_6", "month_7", "month_8", "month_9", "month_10", "month_11", 
                     "month_12", "leap_year_0", "leap_year_1", "leap_year_2", "leap_year_3", "quarter_1", 
                     "quarter_2", "quarter_3", "quarter_4", "log_return_open", "log_return_high", "log_return_low",
                     "log_return_close", "log_volume", "cumulative_volume", "log_cumulative_volume", 
                     "cumulative_vwap", "accumulated_volatility", "normalized_vwap", "buy_bot_1", "sell_bot_1", 
                     "buy_bot_2", "sell_bot_2", "buy_bot_3", "sell_bot_3", "buy_bot_4", "sell_bot_4", "trendup", 
                     "trenddown", "buy_bot_5", "sell_bot_5", "sar", "buy_bot_6", "sell_bot_6", "buy_bot_7", 
                     "sell_bot_7", "buy_bot_8", "sell_bot_8", "rsi_14", "buy_bot_9", "sell_bot_9", "stochastick", 
                     "buy_bot_10", "sell_bot_10", "ema_short", "sma_long", "buy_bot_11", "sell_bot_11", 
                     "bollinger_band_top", "bollinger_band_bottom", "macd_value", "buy_bot_12", "sell_bot_12", 
                     "cci_value", "buy_bot_13", "sell_bot_13", "dmi_plus", "dmi_minus", "adx_value", "buy_bot_14", 
                     "sell_bot_14", "ema_long", "buy_bot_15", "sell_bot_15", "gmma_short", "gmma_long", "buy_bot_16", 
                     "sell_bot_16", "hma_value", "buy_bot_17", "sell_bot_17", "rsi_value", "buy_bot_18", "sell_bot_18", 
                     "buy_bot_19", "sell_bot_19", "buy_bot_20", "sell_bot_20", "buy_bot_21", "sell_bot_21", "donchian_high", 
                     "donchian_low", "buy_bot_22", "sell_bot_22", "buy_bot_23", "sell_bot_23", "buy_bot_24", "sell_bot_24", 
                     "ema_value", "buy_bot_25", "sell_bot_25", "buy_bot_26", "sell_bot_26", "mfim", "buy_bot_27", "sell_bot_27", 
                     "buy_bot_28", "sell_bot_28", "rocm", "buy_bot_29", "sell_bot_29", "ma20", "stochasticd", "buy_bot_30", 
                     "sell_bot_30", "buy_bot_31", "sell_bot_31", "buy_bot_32", "sell_bot_32", "buy_bot_33", "sell_bot_33", 
                     "buy_bot_34", "sell_bot_34", "ema_mid", "buy_bot_35", "sell_bot_35", "buy_bot_36", "sell_bot_36", 
                     "buy_bot_37", "sell_bot_37", "buy_bot_38", "sell_bot_38", "buy_bot_39", "sell_bot_39", "buy_bot_40", 
                     "sell_bot_40", "wvf", "volatilityratio", "buy_bot_41", "sell_bot_41", "buy_bot_42", "sell_bot_42", 
                     "buy_bot_43", "sell_bot_43", "buy_bot_44", "sell_bot_44", "buy_bot_46", "sell_bot_46", "buy_bot_47", 
                     "sell_bot_47", "buy_bot_48", "sell_bot_48", "buy_bot_49", "sell_bot_49", "buy_bot_50", "sell_bot_50", 
                     "pre_expiration", "expiration_day", "post_expiration", "before_holiday", "after_holiday", "avg_obi_1m", 
                     "sum_obi_1m", "min_obi_1m", "max_obi_1m", "std_obi_1m", "avg_spread_1m", "sum_spread_1m", "min_spread_1m", 
                     "max_spread_1m", "std_spread_1m", "avg_ofi_1m", "sum_ofi_1m", "min_ofi_1m", "max_ofi_1m", "std_ofi_1m", 
                     "vwap", "rl_vwap_dv", "avg_mid_price_change_1m", "sum_mid_price_change_1m", "min_mid_price_change_1m", 
                     "max_mid_price_change_1m", "std_mid_price_change_1m", "avg_liquidity_shock_bid_1m", "sum_liquidity_shock_bid_1m",
                     "min_liquidity_shock_bid_1m", "max_liquidity_shock_bid_1m", "std_liquidity_shock_bid_1m", 
                     "avg_liquidity_shock_ask_1m", "sum_liquidity_shock_ask_1m", "min_liquidity_shock_ask_1m", 
                     "max_liquidity_shock_ask_1m", "std_liquidity_shock_ask_1m", "CVD", "mpi_1m", "buy_volume", "sell_volume", 
                     "volume_imbalance", "buy_volume_1m", "sell_volume_1m", "volume_imbalance_1m", "vwap_deviation_1m", "step_0", 
                     "step_1", "step_2", "step_3", "step_4", "step_5", "step_6", "step_7", "step_8", "step_9", "step_10", "step_11", 
                     "step_12", "step_13", "step_14", "step_15", "step_16", "step_17", "open_interest", "F1_OPEN", "F1_CLOSE", 
                     "F1_HIGH", "F1_LOW", "F1_VOLUME", "F1_SHORTTERM", "F1_MEDIUMTERM", "F1_LONGTERM", "F1_ACCDIST", "F1_RSI_14", 
                     "F1_CCI_14", "F1_ROC_C14", "F1_WR_14", "F1_SIGNAL_IBUY", "F1_SIGNAL_TMA", "F1_SIGNAL_MACD", "F1_SIGNAL_Stoch", 
                     "F1_SIGNAL_ADX", "F1_R1", "F1_S1", "F1_R2", "F1_S2", "F1_R3", "F1_S3", "F1_TRSTOP", "F1_TRGPRICE", "VN30_OPEN", 
                     "VN30_CLOSE", "VN30_HIGH", "VN30_LOW", "VN30_VOLUME", "VN30_SHORTTERM", "VN30_MEDIUMTERM", "VN30_LONGTERM", 
                     "VN30_ACCDIST", "VN30_RSI_14", "VN30_CCI_14", "VN30_ROC_C14", "VN30_WR_14", "VN30_SIGNAL_IBUY", "VN30_SIGNAL_TMA", 
                     "VN30_SIGNAL_MACD", "VN30_SIGNAL_Stoch", "VN30_SIGNAL_ADX", "VN30_R1", "VN30_S1", "VN30_R2", "VN30_S2", "VN30_R3", 
                     "VN30_S3", "VN30_TRSTOP", "VN30_TRGPRICE", "VN_OPEN", "VN_CLOSE", "VN_HIGH", "VN_LOW", "VN_VOLUME", "VN_SHORTTERM", 
                     "VN_MEDIUMTERM", "VN_LONGTERM", "VN_ACCDIST", "VN_RSI_14", "VN_CCI_14", "VN_ROC_C14", "VN_WR_14", "VN_SIGNAL_IBUY", 
                     "VN_SIGNAL_TMA", "VN_SIGNAL_MACD", "VN_SIGNAL_Stoch", "VN_SIGNAL_ADX", "VN_R1", "VN_S1", "VN_R2", "VN_S2", "VN_R3", 
                     "VN_S3", "VN_TRSTOP", "VN_TRGPRICE"]
    print (len(features_list))
    # exit()

    feature_mapping = {}
    for feature in features_list:
        feature_mapping[feature] = get_or_create_feature_name(feature)

    for origin_feature, new_feature in feature_mapping.items():
        print(f"{origin_feature} -> {new_feature}")