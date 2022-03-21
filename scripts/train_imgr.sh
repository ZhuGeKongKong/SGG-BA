#!/usr/bin/env bash
export PYTHONPATH=/home/guoyuyu/lib/apex:/home/guoyuyu/lib/cocoapi:/home/guoyuyu/code/scene_graph_gen/SSG-BA:$PYTHONPATH
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=1 #3,4 #,4 #3,4
    export NUM_GUP=1
    echo "TRAINING Image Retrieval"
    MODEL_NAME="transformer_sgdet_Lr1e3_B16_It16/sen2graph_filter1e2_xavier_modelv2/" #"transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4"
    mkdir -p ./checkpoints_best/${MODEL_NAME}/
    python -u \
    tools/image_retrieval_main.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 30 SOLVER.BASE_LR 1e-2 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.STEPS "(10, 25)" SOLVER.VAL_PERIOD 1 \
    SOLVER.CHECKPOINT_PERIOD 5 GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ''\
    MODEL.PRETRAINED_MODEL_CKPT '' \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints_best/${MODEL_NAME}
elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=1 #3,4 #,4 #3,4
    export NUM_GUP=1
    echo "TRAINING Image Retrieval"
    MODEL_NAME="transformer_sgdet_TopDist15_TopBLMaxDist2k_FixBiGraph_lr1e3_B16/sen2graph_filter1e2_xavier_modelv2/" #"transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4"
    mkdir -p ./checkpoints_best/${MODEL_NAME}/
    python -u tools/image_retrieval_main.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 30 SOLVER.BASE_LR 1e-2 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.STEPS "(10, 25)" SOLVER.VAL_PERIOD 1 \
    SOLVER.CHECKPOINT_PERIOD 5 GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ''\
    MODEL.PRETRAINED_MODEL_CKPT '' \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints_best/${MODEL_NAME}
elif [ $1 == "2" ]; then
    export CUDA_VISIBLE_DEVICES=1 #3,4 #,4 #3,4
    export NUM_GUP=1
    echo "TRAINING Image Retrieval"
    MODEL_NAME="transformer_sgdet_dist20_2k_FixPModel_lr1e3_B16_FixCMatDot/sen2graph_filter1e3_xavier_modelv2/" #"transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4"
    mkdir -p ./checkpoints_best/${MODEL_NAME}/
    python -u \
    tools/image_retrieval_main.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 30 SOLVER.BASE_LR 1e-2 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.STEPS "(10, 25)" SOLVER.VAL_PERIOD 1 \
    SOLVER.CHECKPOINT_PERIOD 5 GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ''\
    MODEL.PRETRAINED_MODEL_CKPT '' \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints_best/${MODEL_NAME}
elif [ $1 == "3" ]; then
    export CUDA_VISIBLE_DEVICES=4 #3,4 #,4 #3,4
    export NUM_GUP=1
    echo "TRAINING Image Retrieval"
    MODEL_NAME="transformer_sgdet_Lr1e3_B16_It16/sentence2graph_wobg/" #"transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4"
    python -u tools/image_retrieval_main.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 20 SOLVER.BASE_LR 1e-3 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 1 \
    SOLVER.CHECKPOINT_PERIOD 1 GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ''\
    MODEL.PRETRAINED_MODEL_CKPT '' \
    OUTPUT_DIR ./checkpoints_best/${MODEL_NAME}
fi