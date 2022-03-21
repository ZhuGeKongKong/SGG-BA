#!/usr/bin/env bash
export PYTHONPATH=/home/guoyuyu/lib/apex:/home/guoyuyu/lib/cocoapi:/home/guoyuyu/code/scene_graph_gen/SSG-BA/:$PYTHONPATH
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=0,1  #,4
    export NUM_GUP=2
    echo "Testing Predcls"
    #TransformerSuperPredictor
    #TransformerTransferPredictor
    MODEL_NAME="frequency_predictor"
    #OUTPUT_NAME="transformer_predcls_top20cls_unseen_Mattrans"
    MODEL_PATH=./checkpoints/${MODEL_NAME}
    OUTPUT_PATH=${MODEL_PATH}/inference_rmtop10_filter3_infover/
    mkdir ${OUTPUT_PATH}
    python  -u  -m torch.distributed.launch --master_port 10035 --nproc_per_node=$NUM_GUP \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
            MODEL.ROI_RELATION_HEAD.PREDICTOR FrequencyPredictor \
            MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ${MODEL_PATH}/model_final.pth \
            OUTPUT_DIR ${OUTPUT_PATH} \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0 \
            TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG True \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False
elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=2,3
    export NUM_GUP=2
    echo "Testing Predcls"
    MODEL_NAME="transformer_predcls_dist20_2k_FixPModel_lr1e3_B16_FixCMatDot"
    python  -u  -m torch.distributed.launch --master_port 10036 --nproc_per_node=$NUM_GUP \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
            MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ./checkpoints_best/${MODEL_NAME}/model_final.pth \
            OUTPUT_DIR ./checkpoints_best/${MODEL_NAME} \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0 \
            TEST.ALLOW_LOAD_FROM_CACHE True TEST.VAL_FLAG False;
elif [ $1 == "2" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    export NUM_GUP=2
    echo "Testing SGDet"
    MODEL_NAME="transformer_sgdet_TopDist15_TopBLMaxDist2k_FixBiGraph_lr1e3_B16"
    MODEL_PATH=./checkpoints_best/${MODEL_NAME}
    OUTPUT_PATH=${MODEL_PATH}/inference_rmtop10_filter3_infover/
    mkdir -p ${OUTPUT_PATH}
    python  -u  -m torch.distributed.launch --master_port 10037 --nproc_per_node=$NUM_GUP \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
            MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
            MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS False \
            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ${MODEL_PATH}/model_final.pth \
            OUTPUT_DIR ${OUTPUT_PATH} \
            TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG True \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0 \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True
fi