from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

load_dotenv()

# Cấu hình kết nối PostgreSQL
DB_CONFIG = {
    'dbname': os.getenv("DB_NAME"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'host': os.getenv("DB_HOST"),
    'port': os.getenv("DB_PORT")
}

DB_CONFIG_DEV = {
    'dbname': os.getenv("DB_NAME_2"),
    'user': os.getenv("DB_USER_2"),
    'password': os.getenv("DB_PASSWORD_2"),
    'host': os.getenv("DB_HOST_2"),
    'port': os.getenv("DB_PORT_2")
}

DB_CONFIG_PROD = {
    'dbname': os.getenv("DB_NAME_PROD"),
    'user': os.getenv("DB_USER_PROD"),
    'password': os.getenv("DB_PASSWORD_PROD"),
    'host': os.getenv("DB_HOST_PROD"),
    'port': os.getenv("DB_PORT_PROD")
}

def get_engine():
    return create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")

def get_engine_dev():
    return create_engine(f"postgresql://{DB_CONFIG_DEV['user']}:{DB_CONFIG_DEV['password']}@{DB_CONFIG_DEV['host']}:{DB_CONFIG_DEV['port']}/{DB_CONFIG_DEV['dbname']}")

def get_engine_prod():
    return create_engine(f"postgresql://{DB_CONFIG_PROD['user']}:{DB_CONFIG_PROD['password']}@{DB_CONFIG_PROD['host']}:{DB_CONFIG_PROD['port']}/{DB_CONFIG_PROD['dbname']}")