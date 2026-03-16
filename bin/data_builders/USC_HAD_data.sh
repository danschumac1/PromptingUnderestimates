#!/bin/bash
set -e  # exit immediately if a command fails

# URLs and paths
URL="https://sipi.usc.edu/had/USC-HAD.zip"
RAW_DATA_DIR="data/raw_data"
ZIP_FILE="$RAW_DATA_DIR/USC-HAD.zip"

echo "Creating raw data directory..."
mkdir -p "$RAW_DATA_DIR"

echo "Downloading USC-HAD dataset..."
wget -O "$ZIP_FILE" "$URL"

echo "Unzipping dataset..."
unzip "$ZIP_FILE" -d "$RAW_DATA_DIR"

echo "Removing zip file..."
rm "$ZIP_FILE"

echo "Running cleaning script..."
python ./src/data_management/_clean_USC_HAD.py

echo "Done."
