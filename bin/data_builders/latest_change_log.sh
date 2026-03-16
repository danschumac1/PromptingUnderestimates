#!/bin/bash
# chmod +x ./bin/latest_change_log.sh
# ------------------------------------------------------------
# Walk ./data/sample_features and log .npz modification times
# ------------------------------------------------------------

ROOT_DIR="./data/sample_features"
LOG_DIR="./logs"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/npz_timestamps_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "NPZ file modification log"        >  "$LOG_FILE"
echo "Root directory: $ROOT_DIR"       >> "$LOG_FILE"
echo "Generated at: $(date)"           >> "$LOG_FILE"
echo "--------------------------------------------------" >> "$LOG_FILE"

# Walk directory and log .npz files
find "$ROOT_DIR" -type f -name "*.npz" | while read -r file; do
    mod_time=$(stat -c "%y" "$file")
    printf "%s | %s\n" "$mod_time" "$file" >> "$LOG_FILE"
done

echo "--------------------------------------------------" >> "$LOG_FILE"
echo "Done." >> "$LOG_FILE"

echo "Wrote log to: $LOG_FILE"
