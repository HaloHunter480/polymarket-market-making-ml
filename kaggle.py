import kagglehub

# Download latest version
path = kagglehub.dataset_download("shivaverse/btcusdt-5-minute-ohlc-volume-data-2017-2025")

print("Path to dataset files:", path)