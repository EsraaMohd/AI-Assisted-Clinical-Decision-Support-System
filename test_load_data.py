from src.data.load_data import load_dataset
from config.paths import RAW_DATA_DIR

dataset_path = RAW_DATA_DIR / "hospital_readmissions.csv"

df = load_dataset(dataset_path)

print(df.shape)
print(df.head())