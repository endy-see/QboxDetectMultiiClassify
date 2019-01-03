#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import model.test as model_utils
import sys
from model.nms_wrapper import nms
from model.config import cfg, cfg_from_file
from utils.timer import Timer
import tensorflow as tf
import numpy as np
import os, cv2
import argparse
import pprint
import math

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
from model.bbox_transform import bbox_transform_inv
from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer

def iou(BBGT, bb):
    if len(BBGT) == 0: return 0, 0

    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
            (BBGT[:, 2] - BBGT[:, 0] + 1.) *
            (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

    overlaps = inters / uni
    ovmax = np.max(overlaps)
    jmax = np.argmax(overlaps)

    return ovmax, jmax

class ObjectDetector:
    def __init__(self, rpn_model_path, rcnn_model_path, cfg_file, classes=None, max_image_size=None, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), max_object_per_image=15, conf_thresh=0.3, nms_thresh=0.5, iou_thresh=0.5):
        self.rpn_model_path = rpn_model_path
        self.rcnn_model_path = rcnn_model_path
        self.cfg_file = cfg_file
        if classes is None:
            classes = ['background', 'object']
        self._classes = classes
        self.num_classes = len(self._classes)
        self._max_image_size = max_image_size
        self.num_images = 1
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.iou_thresh = iou_thresh
        self.max_object_per_image = max_object_per_image
        self._feat_stride = [16, ]
        self._feat_compress = [1. / 16., ]
                
        # load network configuration
        cfg_from_file(self.cfg_file)
        # pprint.pprint(cfg)
        self._anchor_scales = cfg.ANCHOR_SCALES
        self._num_scales = len(self._anchor_scales)
        self._anchor_ratios = cfg.ANCHOR_RATIOS
        self._num_ratios = len(self._anchor_ratios)
        self._num_anchors = self._num_scales * self._num_ratios

        with tf.gfile.FastGFile(self.rpn_model_path , 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name="rpn")
                self._images = graph.get_tensor_by_name('rpn/Placeholder:0')
                self._rpn_cls_prob = graph.get_tensor_by_name('rpn/MobilenetV1_2/rpn_cls_prob/transpose_1:0')
                self._rpn_bbox_pred = graph.get_tensor_by_name('rpn/MobilenetV1_2/rpn_bbox_pred/BiasAdd:0')
                self._rpn_feature_map = graph.get_tensor_by_name('rpn/MobilenetV1_1/Conv2d_11_pointwise/Relu6:0')
                self.rpn_graph = graph

        with tf.gfile.FastGFile(self.rcnn_model_path , 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name="rcnn")
                self._rcnn_feature_map = graph.get_tensor_by_name('rcnn/Placeholder_3:0')
                self._rois = graph.get_tensor_by_name('rcnn/Placeholder_4:0')
                self._cls_prob = graph.get_tensor_by_name('rcnn/MobilenetV1_2/cls_prob:0')
                self._bbox_pred = graph.get_tensor_by_name('rcnn/add:0')
                self.rcnn_graph = graph

        # set config
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth=True
        # tfconfig=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
        self.rpn_sess = tf.Session(graph=self.rpn_graph, config=tfconfig) 
        self.rcnn_sess = tf.Session(graph=self.rcnn_graph, config=tfconfig)
        # self.printOperation(self.rcnn_sess)

    def generate_anchors(self, im_info):
        height = int(math.ceil(im_info[0, 0] / np.float32(self._feat_stride[0])))
        width = int(math.ceil(im_info[0, 1] / np.float32(self._feat_stride[0])))
        anchors, anchor_length = generate_anchors_pre(height, width, self._feat_stride, self._anchor_scales, self._anchor_ratios)
   
        anchors = np.reshape(anchors, [-1, 4])
        anchor_length = np.reshape(anchor_length, [-1])
        return anchors, anchor_length

    def _clip_boxes(self, boxes, im_shape):
        """Clip boxes to image boundaries."""
        # x1 >= 0
        boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
        return boxes

    def im_detect(self, im):
        blobs, im_scales = model_utils._get_blobs(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"

        im_blob = blobs['data']
        blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        anchors, anchor_length = self.generate_anchors(blobs['im_info'])

        feed = { 
            self._images: blobs['data']
        }
        rpn_cls_prob, rpn_bbox_pred, rpn_feature_map = self.rpn_sess.run([self._rpn_cls_prob, self._rpn_bbox_pred, self._rpn_feature_map], feed)
        rois, _ = proposal_layer(rpn_cls_prob, rpn_bbox_pred, blobs['im_info'], 'TEST',
                                        self._feat_stride, anchors, self._num_anchors)
        rois = np.reshape(rois, [-1, 5])
        # rpn_scores = np.reshape(rpn_scores, [-1, 1])

        feed = {
            self._rcnn_feature_map: rpn_feature_map,
            self._rois: rois
        }
        scores, bbox_pred = self.rcnn_sess.run([self._cls_prob, self._bbox_pred], feed)
        
        boxes = rois[:, 1:5] / im_scales[0]
        scores = np.reshape(scores, [scores.shape[0], -1])
        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = self._clip_boxes(pred_boxes, im.shape)

        return scores, pred_boxes
    
    def printOperation(self, sess):
        for op in sess.graph.get_operations():
            print(str(op.name))

    def detect(self, image):
        resize_ratio = 1.0
        if self._max_image_size is not None:
            im_width = image.shape[0]
            im_height = image.shape[1]
            resize_ratio = float(self._max_image_size) / max(im_width, im_height)
            width = im_width * resize_ratio
            height = im_height * resize_ratio
            image = cv2.resize(image,(width, height),interpolation=cv2.INTER_CUBIC)

        scores, boxes = self.im_detect(image)
        all_regions = {}
        # process all classes except the background
        for cls_ind in range(1, self.num_classes): 
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            detections = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(detections, self.nms_thresh)
            detections = detections[keep, :]

            regions = []
            for detection in detections:
                overlap, idx = iou(np.asarray(regions), detection)
                if overlap < self.iou_thresh and detection[4] > self.conf_thresh:
                    detection[:4] = detection[:4] * resize_ratio
                    regions.append(detection)
            
            object_class = cls_ind
            if object_class < len(self._classes):
                object_class = self._classes[object_class]
            all_regions[object_class] = regions
           
        return all_regions

def show(im):
    msg = 'press any key to continue'
    cv2.namedWindow(msg, cv2.WINDOW_NORMAL)
    cv2.imshow(msg, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    
    # load text line recongize model
    config_file = sys.argv[1]
    rpn_model_path = sys.argv[2]
    rcnn_model_path = sys.argv[3]


    classes = ['background', 'text']
    detector = ObjectDetector(rpn_model_path, rcnn_model_path, config_file, classes)

    # detector.printOperation()



    while True:
        path = raw_input("Image path (press 0 to exit):")
        if path == '0':
            break
        
        if not os.path.isfile(path):
            print("Image not exists!")
            continue
        
        img = cv2.imread(path)

        # detect text regions
        all_regions = detector.detect(img)
        for key in all_regions.keys():
            boxes = all_regions[key]
            for i in range(len(boxes)):
                print("Boxes {}: {}".format(key, boxes[i]))
                cv2.rectangle(img, (boxes[i][0].astype(np.int32), boxes[i][1].astype(np.int32)), (boxes[i][2].astype(np.int32), boxes[i][3].astype(np.int32)), color=(255, 255, 0), thickness=1)
                
        show(img)