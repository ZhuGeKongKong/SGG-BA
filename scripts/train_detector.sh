#!/usr/bin/env bash
export PYTHONPATH=/home/guoyuyu/lib/apex:/home/guoyuyu/lib/cocoapi:/home/guoyuyu/code/scene_graph_gen/SSG-BA:$PYTHONPATH
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=0,1 #3,4 #,4 #3,4
    export NUM_GUP=2
    MODEL_NAME="pretrained_faster_rcnn_newdict" #"transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4"
    mkdir ./checkpoints_gqa/${MODEL_NAME}/
    echo "TRAINING Detector"
    python -m torch.distributed.launch --master_port 10001 --nproc_per_node=$NUM_GUP \
    tools/detector_pretrain_net.py \
    --config-file "configs/e2e_relation_detector_X_101_32_8_FPN_1x_gqa.yaml" \
    SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
    SOLVER.MAX_ITER 50000  SOLVER.STEPS "(30000, 45000)" \
    SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 \
    MODEL.RELATION_ON False  \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
    OUTPUT_DIR ./checkpoints_gqa/${MODEL_NAME} \
    SOLVER.PRE_VAL False
elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=0,1 #3,4 #,4 #3,4
    export NUM_GUP=2
    MODEL_NAME="pretrained_faster_rcnn" #"transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4"
    mkdir ./checkpoints_gqa/${MODEL_NAME}/
    echo "TRAINING Detector"
    python -m torch.distributed.launch --master_port 10001 --nproc_per_node=$NUM_GUP \
    tools/detector_pretrain_net.py \
    --config-file "configs/e2e_relation_detector_X_101_32_8_FPN_1x_gqa.yaml" \
    SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
    SOLVER.MAX_ITER 100000  SOLVER.STEPS "(60000, 90000)" \
    SOLVER.VAL_PERIOD 4000 SOLVER.CHECKPOINT_PERIOD 4000 \
    MODEL.RELATION_ON False  \
    OUTPUT_DIR ./checkpoints_gqa/${MODEL_NAME} \
    SOLVER.PRE_VAL False
fi