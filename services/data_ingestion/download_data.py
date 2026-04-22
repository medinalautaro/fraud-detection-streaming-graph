import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "mlg-ulb/creditcardfraud"
DOWNLOAD_DIR = "/data"
ZIP_PATH = os.path.join(DOWNLOAD_DIR, "creditcardfraud.zip")
CSV_PATH = os.path.join(DOWNLOAD_DIR, "creditcard.csv")

def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH):
        print(f"[INGESTION] Dataset already available at: {CSV_PATH}")
        return

    print("[INGESTION] Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()

    print(f"[INGESTION] Downloading dataset: {DATASET}")
    api.dataset_download_files(
        DATASET,
        path=DOWNLOAD_DIR,
        unzip=False
    )

    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"Expected zip not found: {ZIP_PATH}")

    print(f"[INGESTION] Extracting: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DOWNLOAD_DIR)

    print(f"[INGESTION] Dataset ready at: {CSV_PATH}")

if __name__ == "__main__":
    main()