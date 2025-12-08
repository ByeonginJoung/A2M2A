export CUDA_VISIBLE_DEVICES=0

AUDIO_FILE=demo_items/trimmed_test0000.wav
OUTPUT_DIR=demo_items
LOG_DIR=logs
CONF_NAME=lstm_msessim_256_mri_melspectogram_baseline_ver0004_multi

python run_pipeline.py \
       --audio_file ${AUDIO_FILE} \
       --output_dir ${OUTPUT_DIR} \
       --log_dir ${LOG_DIR}/${CONF_NAME} \
       --debug_anime
