# Directory containing preprocessed MSCOCO data.
MSCOCO_DIR="${HOME}/im2txt/data/mscoco_5k/BACKUP_TFRecord_train_1600vntk"

# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT="${HOME}/im2txt/data/inception_v3.ckpt"

# Directory to save the model.
MODEL_DIR="${HOME}/im2txt/model_coco5k_1600vntk"

# Build the model.
bazel build -c opt //im2txt/...

LOG="./LOGS/log_train_COCO5k_1600vntk_`date +'%d_%m_%Y__%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")

# Run the training script.
bazel-bin/im2txt/train \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00016" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}" \
  --train_inception=true \
  --number_of_steps=120000