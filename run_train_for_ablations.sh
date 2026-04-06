# A: nothing
# B: prev only
# C: fusion only
# D: do both (ours)

export CUDA_VISIBLE_DEVCIES=0

SUB_NAME=053

#POST_FIXS=('_2' '_6' '_A_2' '_A_6' '_A_10' '_B_2' '_B_6' '_B_10' '_C_2' '_C_6' '_C_10')
POST_FIXS=('_C_10')

for POST_FIX in "${POST_FIXS[@]}"; do

    #CONF_NAME='mri_melspectogram_baseline_ver0004_scene'${SUB_NAME}${POST_FIX}
    CONF_NAME='mri_melspectogram_baseline_ver0004_scene'${SUB_NAME}
    EXP_NAME=lstm_msessim_256_${CONF_NAME}

    DATASET_TYPE="75-speaker"  # can be overridden per data variant

    python train.py --dataset mri \
	   --exp_name ${EXP_NAME} \
	   --sub_name ${SUB_NAME} \
	   --config_name ${CONF_NAME} \
	   --dataset_type ${DATASET_TYPE}
done
