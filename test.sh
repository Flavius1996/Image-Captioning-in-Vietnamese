# Path to checkpoint file or a directory containing checkpoint files. Passing
# a directory will only work if there is also a file named 'checkpoint' which
# lists the available checkpoints in the directory. It will not work if you
# point to a directory with just a copy of a model checkpoint: in that case,
# you will need to pass the checkpoint path explicitly.
#testmodel_1="KranthiGV_trans.ckpt-2000000"
#testmodel_2="model_trans.ckpt-3000000"
#CHECKPOINT_PATH="${HOME}/im2txt/pretrained_models/${testmodel_2}" 

## CHOOSE EN MODEL
#CHECKPOINT_PATH="${HOME}/im2txt/model_coco5k_en_wordcount1" 
#VOCAB_FILE="${HOME}/im2txt/data/mscoco_5k/BACKUP_TFRecord_train_en_wordcount1/word_counts_en_wordcount1.txt"

## CHOOSE GOOGLEVN MODEL
#CHECKPOINT_PATH="${HOME}/im2txt/model_coco5k_googlevn/train" 
#VOCAB_FILE="${HOME}/im2txt/data/mscoco_5k/word_counts_googlevn.txt"

## CHOOSE VN MODEL
CHECKPOINT_PATH="${HOME}/im2txt/model_coco5k_googlecap4plus1_4000tk" 
VOCAB_FILE="${HOME}/im2txt/data/mscoco_5k/BACKUP_TFRecord_train_googlecap4plus1_4000tk/word_counts_googlecap4plus1_4000tk.txt"


# Test only 1 image, not save to json
#IMAGE_FILE="/home/tinhh/im2txt/data/mscoco_5k/raw-data/train2014/COCO5k_train_000000000064.jpg"
#RESULT_JSONFILE=""

# Test whole folder
IMAGE_FILE="${HOME}/im2txt/Images_test/*.jpg"
RESULT_JSONFILE="${HOME}/im2txt/RESULTS/Result_COCO5k_138test_googlecap4plus1_4000tk.json"

# Build the inference binary.
bazel build -c opt //im2txt:run_inference

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
#export CUDA_VISIBLE_DEVICES=""

LOG="./LOGS/log_test_googlecap4plus1_4000tk_`date +'%d_%m_%Y__%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")

# Run inference to generate captions.
bazel-bin/im2txt/run_inference \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_files=${IMAGE_FILE} \
  --output_file=${RESULT_JSONFILE}