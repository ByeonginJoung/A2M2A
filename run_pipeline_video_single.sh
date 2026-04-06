export CUDA_VISIBLE_DEVICES=0

N_SCENES=(30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53)

for N_SCENE in "${N_SCENES[@]}"; do
    FEED_N_SCENE=$(printf "%03d" "$N_SCENE")

    VIDEO_FILE=demo_items/cnn_news00.mp4
    OUTPUT_DIR=demo_items
    LOG_DIR=logs
    CONF_NAME='lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene'${FEED_N_SCENE}

    python run_pipeline_video.py \
	   --video_file ${VIDEO_FILE} \
	   --output_dir ${OUTPUT_DIR} \
	   --log_dir ${LOG_DIR}/${CONF_NAME} \
	   --debug_anime \
	   --use_prev_frame \
	   --concat_outputs

    VIDEO_FILE=demo_items/kbs_news00.mp4

    python run_pipeline_video.py \
	   --video_file ${VIDEO_FILE} \
	   --output_dir ${OUTPUT_DIR} \
	   --log_dir ${LOG_DIR}/${CONF_NAME} \
	   --debug_anime \
	   --use_prev_frame \
	   --concat_outputs

    CNN_BASE='demo_items/comparison_cnn_news00'
    CNN_NAME=${CNN_BASE}'_'${CONF_NAME}'.mp4'
    CNN_OUT='demo_items/test_cnn_'${FEED_N_SCENE}'_news00.mp4'

    KBS_BASE='demo_items/comparison_kbs_news00'
    KBS_NAME=${KBS_BASE}'_'${CONF_NAME}'.mp4'
    KBS_OUT='demo_items/test_kbs_'${FEED_N_SCENE}'_news00.mp4'


    ffmpeg -i ${CNN_NAME} \
	   -vf "crop=iw-mod(iw\,2):ih-mod(ih\,2)" -c:v libx264 -pix_fmt yuv420p \
	   -crf 20 -preset medium -c:a aac -b:a 192k ${CNN_OUT} -y

    ffmpeg -i ${KBS_NAME} \
	   -vf "crop=iw-mod(iw\,2):ih-mod(ih\,2)" -c:v libx264 -pix_fmt yuv420p \
	   -crf 20 -preset medium -c:a aac -b:a 192k ${KBS_OUT} -y
done
