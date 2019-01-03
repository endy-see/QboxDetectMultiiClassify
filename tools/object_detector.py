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
from model.test import im_detect
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

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

CLASSES = ('__background__',
           'text')

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
    def __init__(self, net_name, model_path, cfg_file, num_classes = 2, max_object_per_image=15, conf_thresh=0.3, nms_thresh=0.5, iou_thresh=0.5):
        self.net_name = net_name
        # self.sess = sess
        self.model_path = model_path
        self.cfg_file = cfg_file
        self.num_images = 1
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.iou_thresh = iou_thresh
        self.max_object_per_image = max_object_per_image
        
        # set config
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth=True
        # tfconfig=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
        # init session
        self.sess = tf.Session(config=tfconfig)

        if not os.path.isfile(self.model_path + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(self.model_path + '.meta'))
        
        # load network configuration
        cfg_from_file(self.cfg_file)
        # pprint.pprint(cfg)

        # load network
        if self.net_name == 'vgg16':
            self.net = vgg16(batch_size=1)
        elif self.net_name == 'res50':
            self.net = resnetv1(batch_size=1, num_layers=50)
        elif self.net_name == 'res101':
            self.net = resnetv1(batch_size=1, num_layers=101)
        elif self.net_name == 'res152':
            self.net = resnetv1(batch_size=1, num_layers=152)
        elif self.net_name == 'mobile':
            self.net = mobilenetv1(batch_size=1)
        else:
            raise NotImplementedError

        with self.sess.as_default():
            self.net.create_architecture(self.sess, "TEST", self.num_classes, tag='default',
                            anchor_scales=cfg.ANCHOR_SCALES,
                            anchor_ratios=cfg.ANCHOR_RATIOS)

            saver = tf.train.Saver()
            saver.restore(self.sess, self.model_path)
            #tf.train.write_graph(self.sess.graph_def, './', 'mobilenet_faster_rcnn.pbtxt')
    
    def detect(self, image):
        scores, boxes, rboxes, quadboxes = im_detect(self.sess, self.net, image)
    
        # print('Detection {:d} object proposals'.format( boxes.shape[0]))

        # skip the backgound, only keep the text boxes
        inds = np.where(scores[:, 1] > self.conf_thresh)[0]
        txt_scores = scores[inds, 1]
        txt_boxes = boxes[inds, 4:8]
        txt_rboxes = rboxes[inds, 5:10]
        txt_quadboxes = quadboxes[inds, 8:16]
        txt_dets = np.hstack((txt_boxes, txt_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(txt_dets, self.nms_thresh)
        txt_dets = txt_dets[keep, :]
        txt_rboxes = txt_rboxes[keep, :]
        txt_quadboxes = txt_quadboxes[keep, :]
        txt_dets = np.hstack((txt_dets, txt_rboxes, txt_quadboxes))
        # return txt_dets
        regions = []
        for txt_det in txt_dets:
            overlap, idx = iou(np.asarray(regions), txt_det)
            if overlap < self.iou_thresh:
                regions.append(txt_det)

        return regions
        
    def detectAll(self, image):
        scores, boxes, rboxes, quadboxes = im_detect(self.sess, self.net, image)
        allregions = []
        #for cls_ind, cls in enumerate(CLASSES[1:]):
        for cls_ind in range(self.num_classes-1):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_rboxes = rboxes[:, 5 * cls_ind:5 * (cls_ind + 1)]
            cls_quadboxes = quadboxes[:, 8 * cls_ind:8 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, self.nms_thresh)
            dets = dets[keep, :]
            cls_rboxes = cls_rboxes[keep, :]
            cls_quadboxes = cls_quadboxes[keep, :]
            dets = np.hstack((dets, cls_rboxes, cls_quadboxes))
            regions = []
            for txt_det in dets:
                overlap, idx = iou(np.asarray(regions), txt_det)
                if overlap < self.iou_thresh and txt_det[4] > self.conf_thresh:
                    regions.append(txt_det)
            allregions.append(regions)
        return allregions


def show(im):
    msg = 'press any key to continue'
    cv2.namedWindow(msg, cv2.WINDOW_NORMAL)
    cv2.imshow(msg, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rbox2rect(rbox):
    rect = np.zeros((4,2))
    x1 = rbox[0]
    y1 = rbox[1]
    x2 = rbox[2]
    y2 = rbox[3]
    h = rbox[4]
    if x2==x1:
        theta = math.pi/2
    else:
        theta = math.atan((y2-y1)/(x2-x1))
    dx = h * math.sin(theta)
    dy = h * math.cos(theta)
    
    x3 = x2 - dx
    y3 = y2 + dy
    x4 = x1 - dx
    y4 = y1 + dy

    rect[0] = [x1, y1]
    rect[1] = [x2, y2]
    rect[2] = [x3, y3]
    rect[3] = [x4, y4]
    return rect

# quadr: dilation factor for left, right, up, down
def dilating(box, quadr = [0.00,0.04,0.03,0.03]):
    box = box.reshape([-1,2])
    dilated_box = np.zeros(box.shape).astype(np.float32)
    r = quadr[1]
    dilated_box[1][0] = (1+r)*(box[1][0]- box[0][0])+box[0][0]
    dilated_box[1][1] = (1+r)*(box[1][1]- box[0][1])+box[0][1]
    dilated_box[2][0] = (1+r)*(box[2][0]- box[3][0])+box[3][0]
    dilated_box[2][1] = (1+r)*(box[2][1]- box[3][1])+box[3][1]
    r = quadr[0]
    dilated_box[0][0] = (1+r)*(box[0][0]- box[1][0])+box[1][0]
    dilated_box[0][1] = (1+r)*(box[0][1]- box[1][1])+box[1][1]
    dilated_box[3][0] = (1+r)*(box[3][0]- box[2][0])+box[2][0]
    dilated_box[3][1] = (1+r)*(box[3][1]- box[2][1])+box[2][1]
    box = dilated_box
    r = quadr[2]
    dilated_box[0][0] = (1+r)*(box[0][0]- box[3][0])+box[3][0]
    dilated_box[0][1] = (1+r)*(box[0][1]- box[3][1])+box[3][1]
    dilated_box[1][0] = (1+r)*(box[1][0]- box[2][0])+box[2][0]
    dilated_box[1][1] = (1+r)*(box[1][1]- box[2][1])+box[2][1]
    r = quadr[3]
    dilated_box[2][0] = (1+r)*(box[2][0]- box[1][0])+box[1][0]
    dilated_box[2][1] = (1+r)*(box[2][1]- box[1][1])+box[1][1]
    dilated_box[3][0] = (1+r)*(box[3][0]- box[0][0])+box[0][0]
    dilated_box[3][1] = (1+r)*(box[3][1]- box[0][1])+box[0][1]
    
    dilated_box = dilated_box.reshape([1,-1])[0]
    return dilated_box

def getplateimage(img, boxes):
    dstshape = np.float32([[0,0],[440,0],[440,140],[0,140]])
    M = cv2.getPerspectiveTransform(boxes,dstshape)
    plate_img = cv2.warpPerspective(img,M,(440,140))
    return plate_img

def img_wrapper(img, plate_img, i=0):
    #warpper_shape = (img.shape[0],img.shape[1],img.shape[2])
    #warpper_img = np.zeros(warpper_shape, np.uint8)
    warpper_img = img
    #warpper_img[0:img.shape[0],0:img.shape[1],:] = img
    warpper_img[plate_img.shape[0]*i:plate_img.shape[0]*(i+1),\
                0:plate_img.shape[1],\
                :] = plate_img
    return warpper_img

def showrbox(img, rect, width = 5, color = (0,0,255)):
    cv2.line(img, tuple(rect[0].astype(np.int32)),tuple(rect[1].astype(np.int32)), color=color, thickness=width)
    cv2.line(img, tuple(rect[1].astype(np.int32)),tuple(rect[2].astype(np.int32)), color=color, thickness=width)
    cv2.line(img, tuple(rect[2].astype(np.int32)),tuple(rect[3].astype(np.int32)), color=color, thickness=width)
    cv2.line(img, tuple(rect[3].astype(np.int32)),tuple(rect[0].astype(np.int32)), color=color, thickness=width)
    return 0

# detecting the test.txt images
def plate_test(text_detector):
    #base_path = '/Users/yangyd/dataset/quadplate/'
    base_path = './data/plate/'
    txt_path = base_path+'ImageSets/Main/test.txt'
    img_path_root = base_path+ 'JPEGImages'
    det_path_root = './quaddet_results'
    if not os.path.isdir(det_path_root):
        os.makedirs(det_path_root)
    with open(txt_path,'r') as f:
        for line in f.readlines():
            image_name = line.strip()+'.jpg'
            image_path = os.path.join(img_path_root, image_name)
            detname = os.path.join(det_path_root, image_name)
            print(image_path)
            #prefix = image_path.split('/')[-1]
            #if prefix != '7a3e5b2880b711e78244784f43a62fc5usrcol2017081414115701004.jpg':
            #        continue
                
            img = cv2.imread(image_path)
            org_img = img.copy()
            if img is None:
                print('img None\n')
                continue
            regions = text_detector.detect(img)
            #with open(txtname, 'a') as f:
            for i in range(len(regions)):
                # show bndbox
                boxes = regions[i]
                print(boxes[4])
                cv2.rectangle(img, (boxes[0].astype(np.int32), boxes[1].astype(np.int32)), \
                        (boxes[2].astype(np.int32), boxes[3].astype(np.int32)), color=(255, 255, 0), thickness=5)
                # show rbox
                #rect = rbox2rect(boxes[5:10])
                #showrbox(img, rect)
                # show quadbox
                quadboxes = boxes[10:18]
                dilated_boxes = dilating(quadboxes)
                showrbox(img, dilated_boxes.reshape([-1,2]))
                plate_img = getplateimage(org_img, dilated_boxes.reshape([-1,2]))
                all_img = img_wrapper(img, plate_img, i)
                #cv2.circle(img, tuple([quadboxes[0],quadboxes[1]]), 10 ,color=(255,0,0),thickness=5)
                #points = rect.reshape(1,8)
                #np.savetxt(f, points, fmt='%d', delimiter=',')
            cv2.imwrite(detname, all_img)
# get icdar detection results
def icdar_results(text_detector, root_path):
    for rt, _, files in os.walk(root_path):
        for f in files:
            if f[-4:] == ".jpg" and f[-7:] != "_dt.jpg":
                imgname = f
                prefix = f[0:-4]
                txtname = 'det_'+prefix+'.txt'
                detname = 'det_'+prefix+'_det.jpg'
                imgpath = os.path.join(rt, imgname)
                txtname = os.path.join(rt, txtname)
                detname = os.path.join(rt, detname)
                print(imgpath)
                img = cv2.imread(imgpath)

                regions = text_detector.detect(img)
                with open(txtname, 'a') as f:
                    for i in range(len(regions)):
                        boxes = regions[i]
                        print(boxes[4])
                        rect = rbox2rect(boxes[5:10])
                        #cv2.rectangle(img, (boxes[0].astype(np.int32), boxes[1].astype(np.int32)), \
                        #        (boxes[2].astype(np.int32), boxes[3].astype(np.int32)), color=(255, 255, 0), thickness=1)
                        #showrbox(img, rect)
                        points = rect.reshape(1,8)
                        np.savetxt(f, points, fmt='%d', delimiter=',')
                    #cv2.imwrite(detname, img)
                
def test():
    base_path = '/Users/yangyd/dataset/quadplate/'
    txt_path = base_path+'ImageSets/Main/test.txt'
    img_path_root = base_path+ 'JPEGImages'
    det_path_root = './quaddet_results'
    if not os.path.isdir(det_path_root):
        os.makedirs(det_path_root)
    with open(txt_path,'r') as f:
        for line in f.readlines():
            image_name = line.strip()+'.jpg'
            image_path = os.path.join(img_path_root, image_name)
            detname = os.path.join(det_path_root, image_name)
            print(image_path)
            img = cv2.imread(image_path)
            org_img = img.copy()
            print(img.shape)
            if img is None:
                print('img None\n')
                continue
            boxes = np.array([ 2541.60083008,  1094.93286133,  3023.23242188,  1009.29296875,  3038.2121582,
  1172.20227051,  2562.13378906,  1273.05151367])
            showrbox(img, boxes.reshape([-1,2]),color=(255,0,0))
            dilated_boxes = dilating(boxes)
            showrbox(img, dilated_boxes.reshape([-1,2]),color=(0,0,255))
            plate_img = getplateimage(org_img, dilated_boxes.reshape([-1,2]))
            all_img = img_wrapper(img, plate_img)
            show(all_img)
            break  

if __name__ == '__main__':
    
    text_detector = ObjectDetector('mobile',
                                    './output/default/plate_trainval/default/mobile_faster_rcnn_iter_40000.ckpt',
                                    './experiments/cfgs/plate.yml',
                                    num_classes=2,conf_thresh=0.8, nms_thresh=0.5)
    plate_test(text_detector)
    




