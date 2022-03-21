#!/usr/bin/env bash
export PYTHONPATH=/home/guoyuyu/lib/apex:/home/guoyuyu/lib/cocoapi:/home/guoyuyu/code/scene_graph_gen/SSG-BA:$PYTHONPATH
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=0,1 #3,4 #,4 #3,4
    export NUM_GUP=2
    echo "TRAINING Predcls"
    MODEL_NAME="transformer_predcls_tde_blru" #"transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4"
    MODEL_PATH=./checkpoints/${MODEL_NAME}
    OUTPUT_PATH=${MODEL_PATH}/test_TDE/
    echo ${OUTPUT_PATH}
    mkdir -p ${OUTPUT_PATH}
    python -u -m torch.distributed.launch --master_port 10051 --nproc_per_node=$NUM_GUP \
    tools/relation_test_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
    MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER transformer \
    MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE \
    MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 16000 SOLVER.BASE_LR 1e-3 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 4000 GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.PRETRAINED_MODEL_CKPT "" \
    MODEL.WEIGHT ${MODEL_PATH}/model_final.pth \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
    OUTPUT_DIR ${OUTPUT_PATH}
elif [ $1 == "6" ]; then
    export CUDA_VISIBLE_DEVICES=2,3 #3,4 #,4 #3,4
    export NUM_GUP=2
    echo "TRAINING Predcls"
    MODEL_NAME="transformer_predcls_TopDist15_Dist2k_SOLap_lr1e3_B16" #"transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4"
    mkdir ./checkpoints/${MODEL_NAME}/
    cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/data/datasets/visual_genome.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/inference.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py ./checkpoints/${MODEL_NAME}/
    #./checkpoints/transformer_bias_keeptop10/model_final.pth
    python -u -m torch.distributed.launch --master_port 10050 --nproc_per_node=$NUM_GUP \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 16000 SOLVER.BASE_LR 1e-3 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 4000 GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.PRETRAINED_MODEL_CKPT ./checkpoints_best/transformer_predcls_float32_epoch16_batch16/model_final.pth \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME}
elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=4,5 #3,4 #,4 #3,4
    export NUM_GUP=2
    echo "TRAINING Transformer SGcls"
    MODEL_NAME="transformer_sgcls_TopDist15_TopBLMaxDist2k_FixPModel_lr1e3_B16"
    mkdir ./checkpoints/${MODEL_NAME}/
    cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/data/datasets/visual_genome.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/layers/gcn/gcn_layers.py ./checkpoints/${MODEL_NAME}/
    python -u -m torch.distributed.launch --master_port 10020 --nproc_per_node=$NUM_GUP \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 16000 SOLVER.BASE_LR 1e-3 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.PRETRAINED_MODEL_CKPT ./checkpoints_best/transformer_sgcls_Lr1e3_B16_It16/model_final.pth \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME}
elif [ $1 == "2" ]; then
    export CUDA_VISIBLE_DEVICES=4,5 #3,4 #,4 #3,4
    export NUM_GUP=2
    echo "TRAINING SGdet"
    MODEL_NAME="transformer_sgdet_TopDist15_TopBLMaxDist2k_FixBiGraph_lr1e3_B16"
    mkdir ./checkpoints/${MODEL_NAME}/
    cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/data/datasets/visual_genome.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/layers/gcn/gcn_layers.py ./checkpoints/${MODEL_NAME}/
    python -u -m torch.distributed.launch --master_port 10021 --nproc_per_node=$NUM_GUP \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 16000 SOLVER.BASE_LR 1e-3 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.PRETRAINED_MODEL_CKPT ./checkpoints_best/transformer_sgdet_Lr1e3_B16_It16/model_final.pth \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME}
fi