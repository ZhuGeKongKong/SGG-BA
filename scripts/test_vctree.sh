#!/usr/bin/env bash
export PYTHONPATH=/home/guoyuyu/lib/apex:/home/guoyuyu/lib/cocoapi:/home/guoyuyu/code/scene_graph_gen/SSG-BA/:$PYTHONPATH
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    export NUM_GUP=2
    echo "Testing Predcls"
    MODEL_NAME="vctree_predcls_TopDist15_TopBLMaxDist2k_FixPModel_lr1e3_B16_FBLMCMat"
    python  -u  -m torch.distributed.launch --master_port 10035 --nproc_per_node=$NUM_GUP \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_vctree.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
            MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_0014000.pth \
            OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
            TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG False \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0;
elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=2,3
    export NUM_GUP=2
    echo "Testing SGCls"
    MODEL_NAME="vctree_sgcls_TopDist15_TopBLMaxDist2k_FixPModel_lr1e3_B16_FBLMCMat"
    python  -u  -m torch.distributed.launch --master_port 10036 --nproc_per_node=$NUM_GUP \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_vctree.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
            MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
            MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_0014000.pth \
            OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
            TEST.ALLOW_LOAD_FROM_CACHE True TEST.VAL_FLAG False \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0;

elif [ $1 == "2" ]; then
    export CUDA_VISIBLE_DEVICES=1
    export NUM_GUP=1
    echo "Testing SGDet"
    MODEL_NAME="vctree_sgdet_TopDist15_TopBLMaxDist2k_FixBiGraph_lr1e3_B16"
    python  -u  -m torch.distributed.launch --master_port 10036 --nproc_per_node=$NUM_GUP \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_vctree.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
            MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_final.pth \
            OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
            TEST.ALLOW_LOAD_FROM_CACHE True TEST.VAL_FLAG False \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0;
fi