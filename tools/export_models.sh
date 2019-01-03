#!/bin/bash

# Split, compress and export faster-rcnn model file, usage:
# ./export_model.sh NET_NAME INPUT_MODEL_PATH MODEL_CONFIG_FILE CLASSES_NUM CHANNEL_NUM FOLDER_PATH_TO_SAVE_THE_EXPORTED_MODEL
# example:
# ./export_models.sh mobile /home/linkface/tf-faster-rcnn/output/mobile/DriverText_trainval/mobile_0.25/mobile_faster_rcnn_iter_70000.ckpt \
#       /home/linkface/tf-faster-rcnn/experiments/cfgs/text.yml 4 128 /tmp/

# ./export_models.sh res50 /home/linkface/tf-faster-rcnn/output/res50/DriverText_trainval/default/res50_faster_rcnn_iter_70000.ckpt \
#       /home/linkface/tf-faster-rcnn/experiments/cfgs/text.yml 4 1024 /tmp/

# Six files will be exported:
# 1. the rpn graph descript file named: '{OUTPUT_PATH}{MODEL_NAME}_rpn.pbtxt'
# 2. the rcnn graph descript file named: '{OUTPUT_PATH}{MODEL_NAME}_rcnn.pbtxt'
# 3. freezed graph rpn model file: '{OUTPUT_PATH}{MODEL_NAME}_rpn.pb' 
# 4. optimized and compressed rpn model file: 'compressed_{OUTPUT_PATH}{MODEL_NAME}_rpn.pb' 
# 5. freezed graph rcnn model file: '{OUTPUT_PATH}{MODEL_NAME}_rcnn.pb' 
# 6. optimized and compressed rcnn model file: 'compressed_{OUTPUT_PATH}{MODEL_NAME}_rcnn.pb' 


NET_NAME=$1
INPUT_MODEL_PATH=$2
CONFIG_FILE_PATH=$3
CLASSES_NUM=$4
CHANNEL_NUM=$5
OUTPUT_PATH=$6

MODEL_NAME=${NET_NAME}_faster_rcnn


if [ ${NET_NAME} = 'mobile' ]
then
    RPN_OUTPUT_NODE_NAMES="MobilenetV1_1/Conv2d_11_pointwise/Relu6,MobilenetV1_2/rpn_cls_prob/transpose_1,MobilenetV1_2/rpn_bbox_pred/BiasAdd"
    RCNN_OUTPUT_NODE_NAMES="MobilenetV1_2/cls_prob,add_2"
    # RCNN_OUTPUT_NODE_NAMES="MobilenetV1_2/cls_prob,add,add_1,add_2"
else
    RPN_OUTPUT_NODE_NAMES="resnet_v1_50_2/block3/unit_6/bottleneck_v1/Relu,resnet_v1_50_3/rpn_cls_prob/transpose_1,resnet_v1_50_3/rpn_bbox_pred/BiasAdd"
    RCNN_OUTPUT_NODE_NAMES="resnet_v1_50_2/cls_prob,add"
    CHANNEL_NUM=1024
fi

# export model graph descript text file
python ./export_graph.py EXPORT ${NET_NAME} ${INPUT_MODEL_PATH} ${CONFIG_FILE_PATH} ${CLASSES_NUM} ${CHANNEL_NUM} ${OUTPUT_PATH} ${MODEL_NAME}_rcnn.pbtxt
python ./export_graph.py TEST ${NET_NAME} ${INPUT_MODEL_PATH} ${CONFIG_FILE_PATH} ${CLASSES_NUM} ${CHANNEL_NUM} ${OUTPUT_PATH} ${MODEL_NAME}_rpn.pbtxt


RPN_GRAPH_TXT_FILE="${OUTPUT_PATH}${MODEL_NAME}_rpn.pbtxt"
RCNN_GRAPH_TXT_FILE="${OUTPUT_PATH}${MODEL_NAME}_rcnn.pbtxt"
OUTPUT_RPN_MODEL_FILE="${OUTPUT_PATH}${MODEL_NAME}_rpn.pb"
OUTPUT_RCNN_MODEL_FILE="${OUTPUT_PATH}${MODEL_NAME}_rcnn.pb"

# split the faster rcnn model into tow sub model: rpn and rcnn
# freeze the rpn model graph
/Users/lairf/AI/tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph  \
    --input_graph=${RPN_GRAPH_TXT_FILE} \
    --input_checkpoint=${INPUT_MODEL_PATH} \
    --output_graph=${OUTPUT_RPN_MODEL_FILE} \
    --output_node_names=${RPN_OUTPUT_NODE_NAMES}

# freeze the rcnn model graph
/Users/lairf/AI/tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph \
    --input_graph=${RCNN_GRAPH_TXT_FILE} \
    --input_checkpoint=${INPUT_MODEL_PATH} \
    --output_graph=${OUTPUT_RCNN_MODEL_FILE} \
    --output_node_names=${RCNN_OUTPUT_NODE_NAMES}


# compress the rpn model graph
COMPRESSED_RPN_MODEL_FILE="${OUTPUT_PATH}/compressed_${MODEL_NAME}_rpn.pb"
COMPRESSED_RCNN_MODEL_FILE="${OUTPUT_PATH}/compressed_${MODEL_NAME}_rcnn.pb"

/Users/lairf/AI/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${OUTPUT_RPN_MODEL_FILE} \
    --out_graph=${COMPRESSED_RPN_MODEL_FILE} \
    --inputs='Placeholder' \
    --outputs=${RPN_OUTPUT_NODE_NAMES} \
    --transforms='
        quantize_weights
        sort_by_execution_order'

/Users/lairf/AI/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${OUTPUT_RCNN_MODEL_FILE} \
    --out_graph=${COMPRESSED_RCNN_MODEL_FILE} \
    --inputs='Placeholder_3,Placeholder_4' \
    --outputs=${RCNN_OUTPUT_NODE_NAMES} \
    --transforms='
        quantize_weights
        sort_by_execution_order'