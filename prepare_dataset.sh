# Location to save the MSCOCO data.
MSCOCO_DIR="${HOME}/im2txt/data/mscoco_5k/BACKUP_TFRecord_train_1600vntk"

# Build the preprocessing script.
bazel build //im2txt:preprocess_VNcap

LOG="./LOGS/log_prepare_1600vntk_`date +'%d_%m_%Y__%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")

# Run the preprocessing script.
bazel-bin/im2txt/preprocess_VNcap "${MSCOCO_DIR}"