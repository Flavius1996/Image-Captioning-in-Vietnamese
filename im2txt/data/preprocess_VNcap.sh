OUTPUT_DIR="${1%/}"
mkdir -p "${OUTPUT_DIR}"

SCRATCH_DIR="${OUTPUT_DIR}/../raw-data"
CURRENT_DIR=$(pwd)
WORK_DIR="$0.runfiles/im2txt/im2txt"
TRAIN_IMAGE_DIR="${SCRATCH_DIR}/train2014"

TRAIN_CAPTIONS_FILE="${SCRATCH_DIR}/annotations/captions_COCO5k_train_1600vntk.json"

# Build TFRecords of the image data.
cd "${CURRENT_DIR}"
BUILD_SCRIPT="${WORK_DIR}/build_VNcap_data"

"${BUILD_SCRIPT}" \
  --train_image_dir="${TRAIN_IMAGE_DIR}" \
  --train_captions_file="${TRAIN_CAPTIONS_FILE}" \
  --output_dir="${OUTPUT_DIR}" \
  --word_counts_output_file="${OUTPUT_DIR}/word_counts_1600vntk.txt" \
