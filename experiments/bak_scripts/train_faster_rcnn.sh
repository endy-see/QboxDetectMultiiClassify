#!/bin/bash

set -x
set -e

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
NET_CFG=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  ICDAR2015)
    TRAIN_IMDB="ICDAR2015_trainval"
    TEST_IMDB="ICDAR2015_test"
    STEPSIZE="[50000,100000]"
    ITERS=150000
    RATIOS="[0.5,1,2]"
    ANCHORS="[4,8,16,32]"
    #RATIOS="[0.1,0.2,0.3,10,5,3]"
    #ANCHORS="[2,4,6]"
    ;;
  plate)
    TRAIN_IMDB="plate_trainval"
    TEST_IMDB="plate_test"
    STEPSIZE="[20000,40000]"
    ITERS=50000
    RATIOS="[0.4,1]"
    ANCHORS="[4,8,16,32]"
    #RATIOS="[0.1,0.2,0.3,10,5,3]"
    #ANCHORS="[2,4,6]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
    NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x
if [ ! -f ${NET_FINAL}.index ]; then
    if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
            --weight data/imagenet_weights/${NET}.ckpt \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg experiments/cfgs/${NET_CFG}.yml \
            --tag ${EXTRA_ARGS_SLUG} \
            --net ${NET} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
    else
        CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
            --weight data/imagenet_weights/${NET}.ckpt \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg experiments/cfgs/${NET_CFG}.yml \
            --net ${NET} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
    fi
fi

./experiments/scripts/test_faster_rcnn.sh $@
