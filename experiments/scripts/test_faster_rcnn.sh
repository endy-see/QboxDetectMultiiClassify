#!/bin/bash

set -x
set -e

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
  DML)
    TRAIN_IMDB="DML_trainval"
    TEST_IMDB="DML_test"
    STEPSIZE="[10000,20000,50000]"
    ITERS=25000
    RATIOS="[0.5,0.68,1,1.48,2]"
    ANCHORS="[1,2,4,8]"
    ;;
  SINO)
    TRAIN_IMDB="SINO_trainval"
    TEST_IMDB="SINO_test"
    STEPSIZE="[30000,60000,100000]"
    ITERS=150000
    RATIOS="[0.1,0.2,0.3,0.5,1,1.2]"
    ANCHORS="[1,2,4,8]"
    ;;
  AuditText)
    TRAIN_IMDB="AuditText_trainval"
    TEST_IMDB="AuditText_test"
    STEPSIZE="[150000,300000,600000]"
    #ITERS=1200000
    ITERS=10000
    RATIOS="[0.1,0.2,0.3,0.5,1,1.2]"
    ANCHORS="[2,4,8]"
    ;;
  UnionPayLogoV2)
    TRAIN_IMDB="UnionPayLogoV2_trainval"
    TEST_IMDB="UnionPayLogoV2_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=40000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoV3)
    TRAIN_IMDB="UnionPayLogoV3_trainval"
    TEST_IMDB="UnionPayLogoV3_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=12000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoV4)
    TRAIN_IMDB="UnionPayLogoV4_trainval"
    TEST_IMDB="UnionPayLogoV4_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=5000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoV5)
    TRAIN_IMDB="UnionPayLogoV5_trainval"
    TEST_IMDB="UnionPayLogoV5_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=38000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoV6)
    TRAIN_IMDB="UnionPayLogoV6_trainval"
    TEST_IMDB="UnionPayLogoV6_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=8000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoV7)
    TRAIN_IMDB="UnionPayLogoV7_trainval"
    TEST_IMDB="UnionPayLogoV7_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=52000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoV8)
    TRAIN_IMDB="UnionPayLogoV8_trainval"
    TEST_IMDB="UnionPayLogoV8_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=20000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoV9)
    TRAIN_IMDB="UnionPayLogoV9_trainval"
    TEST_IMDB="UnionPayLogoV9_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=51000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoV10)
    TRAIN_IMDB="UnionPayLogoV10_trainval"
    TEST_IMDB="UnionPayLogoV10_test"
    STEPSIZE="[13000, 20000, 50000]"
    ITERS=59000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoV11)
    TRAIN_IMDB="UnionPayLogoV11_trainval"
    TEST_IMDB="UnionPayLogoV11_test"
    STEPSIZE="[13000, 20000, 50000]"
    ITERS=14000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoV12)
    TRAIN_IMDB="UnionPayLogoV12_trainval"
    TEST_IMDB="UnionPayLogoV12_test"
    STEPSIZE="[13000, 20000, 50000]"
    ITERS=52000
    RATIOS="[0.68,1,1.48]"
    #ANCHORS="[1,2,4,8,16]"
    ANCHORS="[3,5,7,8,9]"
    ;;
  UnionPayLogoV13)
    TRAIN_IMDB="UnionPayLogoV13_trainval"
    TEST_IMDB="UnionPayLogoV13_test"
    STEPSIZE="[13000, 20000, 50000]"
    ITERS=32000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoV14)
    TRAIN_IMDB="UnionPayLogoV14_trainval"
    TEST_IMDB="UnionPayLogoV14_test"
    STEPSIZE="[13000, 20000, 50000]"
    ITERS=100000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoV15)
    TRAIN_IMDB="UnionPayLogoV15_trainval"
    TEST_IMDB="UnionPayLogoV15_test"
    STEPSIZE="[20000, 40000, 80000]"
    ITERS=72000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[3,5,7,8,9]"
    ;;
  UnionPayLogoOthers)
    TRAIN_IMDB="UnionPayLogoOthers_trainval"
    TEST_IMDB="UnionPayLogoOthers_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=95000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoOthersV2)
    TRAIN_IMDB="UnionPayLogoOthersV2_trainval"
    TEST_IMDB="UnionPayLogoOthersV2_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=70000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoOthersV3)
    TRAIN_IMDB="UnionPayLogoOthersV3_trainval"
    TEST_IMDB="UnionPayLogoOthersV3_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=14000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoOthersV4)
    TRAIN_IMDB="UnionPayLogoOthersV4_trainval"
    TEST_IMDB="UnionPayLogoOthersV4_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=66000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogoOthersV5)
    TRAIN_IMDB="UnionPayLogoOthersV5_trainval"
    TEST_IMDB="UnionPayLogoOthersV5_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=30000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  UnionPayLogo)
    TRAIN_IMDB="UnionPayLogo_trainval"
    TEST_IMDB="UnionPayLogo_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=200000
    RATIOS="[0.68,1,1.48]"
    # ANCHORS="[6,10,16,24]"
    ANCHORS="[1,2,4,8,16]"
    ;;
  Audit)
    TRAIN_IMDB="Audit_trainval"
    TEST_IMDB="Audit_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=320000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[6,10,16,24]"
    ;;
  CreditReport)
    TRAIN_IMDB="CreditReport_trainval"
    TEST_IMDB="CreditReport_test"
    STEPSIZE="[40000,80000,160000]"
    ITERS=265000
    RATIOS="[0.03,0.05,0.08,0.2,0.4,0.8]"
    #ANCHORS="[1,2,4,8,16,32]"
    ANCHORS="[0.75,1,2,4,8,12,16]"
    ;;
 CardLocation)
    TRAIN_IMDB="CardLocation_trainval"
    TEST_IMDB="CardLocation_test"
    STEPSIZE="[40000,80000,160000]"
    ITERS=90000
    RATIOS="[0.68,1.0,1.48]"
    ANCHORS="[4,8,16]"
    ;;
  IdAddress)
    TRAIN_IMDB="IdAddress_trainval"
    TEST_IMDB="IdAddress_test"
    STEPSIZE="[40000,80000,160000]"
    ITERS=30000
    RATIOS="[0.05,0.08,0.2,0.4,0.8]"
    ANCHORS="[1,2,3,4]"
    ;;
  VATInvoice)
    TRAIN_IMDB="VATInvoice_trainval"
    TEST_IMDB="VATInvoice_test"
    STEPSIZE="[40000,80000,160000]"
    ITERS=100000
    RATIOS="[0.08,0.2,0.4,0.8]"
    ANCHORS="[1,2,4,8]"
    ;;
  VPAllText)
    TRAIN_IMDB="VPAllText_trainval"
    TEST_IMDB="VPAllText_test"
    STEPSIZE="[40000,80000,160000]"
    ITERS=300000
    RATIOS="[0.1,0.2,0.3]"
    ANCHORS="[2,4,8,16]"
    ;;
  VPNewText)
    TRAIN_IMDB="VPNewText_trainval"
    TEST_IMDB="VPNewText_test"
    STEPSIZE="[40000,80000,160000]"
    ITERS=200000
    RATIOS="[0.1,0.2,0.3]"
    ANCHORS="[2,4,8,16]"
    ;;
  VPText)
    TRAIN_IMDB="VPText_trainval"
    TEST_IMDB="VPText_test"
    STEPSIZE="[40000,80000,160000]"
    ITERS=200000
    RATIOS="[0.1,0.2,0.3]"
    ANCHORS="[2,4,8,16]"
    ;;
  CardText)
    TRAIN_IMDB="CardText_trainval"
    TEST_IMDB="CardText_test"
    STEPSIZE="[40000,80000,160000]"
    ITERS=615000
    RATIOS="[0.1,0.2,0.3,0.5,1]"
    ANCHORS="[2,4,8]"
    ;;
  MultiLicense)
    TRAIN_IMDB="MultiLicense_trainval"
    TEST_IMDB="MultiLicense_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=100000
    RATIOS="[0.68,1,1.48]"
    ANCHORS="[6,10,16]"
    ;;
  EarTag)
    TRAIN_IMDB="EarTag_trainval"
    TEST_IMDB="EarTag_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=100000
    RATIOS="[0.6,1]"
    ANCHORS="[1,2]"
    ;;
  BussinessLicense2017)
    TRAIN_IMDB="BussinessLicense2017_trainval"
    TEST_IMDB="BussinessLicense2017_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=70000
    ANCHORS="[2,4,8]"
    RATIOS="[0.1,0.2,0.3]"
    ;;
  ExpressLicense)
    TRAIN_IMDB="ExpressLicense_trainval"
    TEST_IMDB="ExpressLicense_test"
    STEPSIZE="[10000,30000,50000]"
    ITERS=13000
    RATIOS="[0.25,1,4]"
    ANCHORS="[6,10,16]"
    ;;
  Plate)
    TRAIN_IMDB="Plate_trainval"
    TEST_IMDB="Plate_test"
    STEPSIZE="[20000,40000,60000]"
    ITERS=100000
    ANCHORS="[2,4,8]"
    RATIOS="[0.33,0.5,1]"
    ;;
  IdCard)
    TRAIN_IMDB="IdCard_trainval"
    TEST_IMDB="IdCard_test"
    STEPSIZE="[100000,200000]"
    ITERS=100000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.62,1,1.6]"
    ;;
  IdText)
    TRAIN_IMDB="IdText_trainval"
    TEST_IMDB="IdText_test"
    STEPSIZE="[10000]"
    ITERS=200000
    ANCHORS="[2,4,8]"
    RATIOS="[0.1,0.2,0.3,0.5,1]"
    ;;
  Cow)
    TRAIN_IMDB="Cow_trainval"
    TEST_IMDB="Cow_test"
    STEPSIZE=[50000]
    ITERS=70000
    ANCHORS="[4,8,16]"
    RATIOS="[0.5,1,2]"
    ;;
  Head)
    TRAIN_IMDB="Head_trainval"
    TEST_IMDB="Head_test"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[2,4,8]"
    RATIOS="[0.5,1,2]"
    ;;
  ExpressText)
    TRAIN_IMDB="ExpressText_trainval"
    TEST_IMDB="ExpressText_test"
    STEPSIZE="[30000,50000,70000]"
    ITERS=90000
    ANCHORS="[2,3,4]"
    RATIOS="[0.1,1.0,10]"
    ;;
  VdibLicense)
    TRAIN_IMDB="VdibLicense_trainval"
    TEST_IMDB="VdibLicense_test"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[6,10,16]"
    RATIOS="[0.68,1,1.48]"
    ;;
  DriverLicense)
    TRAIN_IMDB="DriverLicense_trainval"
    TEST_IMDB="DriverLicense_test"
    STEPSIZE=50000
    ITERS=100000
    ANCHORS="[6,10,16,30]"
    RATIOS="[0.68,1,1.48]"
    ;;
  DriverText)
    TRAIN_IMDB="DriverText_trainval"
    TEST_IMDB="DriverText_test"
    STEPSIZE=50000
    ITERS=150000
    ANCHORS="[2,3]"
    RATIOS="[0.1,0.2,0.3]"
    ;;
  VehicleLicense)
    TRAIN_IMDB="VehicleLicense_trainval"
    TEST_IMDB="VehicleLicense_test"
    STEPSIZE="[10000,20000]"
    ITERS=100000
    ANCHORS="[4,8,16,30]"
    RATIOS="[0.68,1,1.48]"
    ;;
  VehicleText)
    TRAIN_IMDB="VehicleText_trainval"
    TEST_IMDB="VehicleText_test"
    STEPSIZE=50000
    ITERS=70000
    ANCHORS="[2,3]"
    RATIOS="[0.1,0.2,0.3]"
    ;;
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=110000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/test_${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
 NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
 # NET_FINAL=output/${NET}/AuditText_trainval/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
 # NET_FINAL=output/${NET}/Audit_trainval_old/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
 # NET_FINAL=output/${NET}/Audit_trainval_new/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
 # NET_FINAL=output/${NET}/Audit_trainval/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET_CFG}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET_CFG}.yml \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ${EXTRA_ARGS}
fi

