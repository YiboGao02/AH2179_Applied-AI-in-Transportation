import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

CORR_FEATURES = [
    "SPEED_MS_AVG",
    "FLOW",
    "Hour",
    "Is_Peak_Morning",
    "Is_Weekend",
    "Portal_Flow_Mean",
    "Portal_Speed_Mean"
]


def load_dataset(csv_path):
    df = pd.read_csv(csv_path, sep=";")
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
    return df


def engineer_features(df):
    df = df.copy()
    time_dt = pd.to_datetime(df["Time"], format="%H:%M:%S")
    df["Day"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.month_name()
    df["Hour"] = time_dt.dt.hour
    df["Minute"] = time_dt.dt.minute
    df["Minute_Of_Day"] = df["Hour"] * 60 + df["Minute"]

    day_mapping = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
        "Sunday": 7
    }
    df["Day_ID"] = df["Day"].map(day_mapping)

    month_mapping = {
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12
    }
    df["Month_ID"] = df["Month"].map(month_mapping)

    df["Is_Weekend"] = df["Day_ID"].isin([6, 7]).astype(int)
    df["Is_Peak_Morning"] = df["Minute_Of_Day"].between(7 * 60 + 30, 8 * 60 + 30).astype(int)
    df["Daily_Sin"] = np.sin(2 * np.pi * df["Minute_Of_Day"] / 1440.0)
    df["Daily_Cos"] = np.cos(2 * np.pi * df["Minute_Of_Day"] / 1440.0)

    portal_numeric = df["PORTAL"].astype(str).str.extract(r"(\d+[\.,]?\d*)", expand=False)
    portal_numeric = portal_numeric.str.replace("[^0-9]", "", regex=True)
    df["Portal_Pos"] = pd.to_numeric(portal_numeric, errors="coerce")

    df["Portal_Flow_Mean"] = df.groupby("PORTAL")["FLOW"].transform("mean")
    df["Portal_Speed_Mean"] = df.groupby("PORTAL")["SPEED_MS_AVG"].transform("mean")
    return df


def prepare_model_dataframe(df):
    return df.drop(columns=["Day", "Month"]).copy()


def compute_compact_correlation(dataframe, features=None):
    if features is None:
        features = CORR_FEATURES
    return dataframe[features].corr()


def plot_correlation_heatmap(corr_matrix, title="Compact Correlation Matrix", figsize=(8, 6)):
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="plasma", square=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()
