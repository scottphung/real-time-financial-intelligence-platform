import os
import pandas as pd
from datetime import datetime


def write_to_parquet(data: dict):
    dt = datetime.now()

    base_path = f"data_lake/crypto/year={dt.year}/month={dt.month}/day={dt.day}"
    os.makedirs(base_path, exist_ok=True)

    file_path = f"{base_path}/crypto.parquet"

    df = pd.DataFrame([data])

    # append-safe write
    if os.path.exists(file_path):
        existing = pd.read_parquet(file_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_parquet(file_path, index=False)

    print(f"💾 Stored → {file_path}")