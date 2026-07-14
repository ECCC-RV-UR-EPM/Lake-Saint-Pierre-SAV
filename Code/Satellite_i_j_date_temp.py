import pandas as pd
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================
# ============================================
base_dir = os.path.abspath(os.path.join(HERE, "..", "Data", "01_temperature", "Temp_prediction_LSP"))

csv_file = os.path.join(base_dir, "LSP_LSWT_200m_rev02_2000-2024_satellite.csv")

output_csv = os.path.join(base_dir, "Temp_training_daily.csv")

if os.path.exists(output_csv):
    os.remove(output_csv)

# ============================================
# ============================================
print(" START PROCESSING...")

chunksize = 1_000_000
first_chunk = True
total_rows = 0

for i, chunk in enumerate(pd.read_csv(
        csv_file,
        chunksize=chunksize,
        usecols=[5,6,7,8],
        names=["i","j","date","Water_temp"],
        header=0,
        on_bad_lines="skip"
)):

    chunk["date"] = chunk["date"].astype(str).str.strip()
    chunk["date"] = chunk["date"].str.replace(",", "", regex=False)
    chunk["date"] = chunk["date"].str.replace('"', "", regex=False)

    d1 = pd.to_datetime(chunk["date"], format="%d/%m/%Y", errors="coerce")
    d2 = pd.to_datetime(chunk["date"], format="%Y-%m-%d", errors="coerce")

    chunk["Date"] = d1.fillna(d2)

    mask = chunk["Date"].isna()
    chunk.loc[mask, "Date"] = pd.to_datetime(chunk.loc[mask, "date"], errors="coerce")

    chunk = chunk.dropna(subset=["Date", "Water_temp"]).copy()

    chunk["Date"] = chunk["Date"].dt.strftime("%Y-%m-%d")
    chunk["i"] = chunk["i"].astype(int)
    chunk["j"] = chunk["j"].astype(int)

    chunk = chunk[["i","j","Date","Water_temp"]]

    chunk.to_csv(
        output_csv,
        mode="a",
        index=False,
        header=first_chunk
    )

    first_chunk = False
    total_rows += len(chunk)

    print(f" chunk {i+1} saved | rows = {len(chunk)} | total = {total_rows}")

print("\n===================================")
print(f" DONE: {total_rows}")
print(f" : {output_csv}")
print("===================================")
