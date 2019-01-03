# -*- coding=utf-8 -*-
#!/usr/bin/env python

# --------------------------------------------------------
# detect plate in several dirs
# keep the structure of dirs
# --------------------------------------------------------

"""

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
import os, cv2, io
import argparse
import pprint
import math

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
from shared_mc import shared_mc

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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
    dstshape = np.float32([[0,0],[113,0],[113,36],[0,36]])
    M = cv2.getPerspectiveTransform(boxes,dstshape)
    plate_img = cv2.warpPerspective(img,M,(113,36))
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


def labelfromtxt(label_path):
    quadboxes = []
    with io.open(label_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        line_num = len(lines)
        for line in lines:
            data = line.split(',')[0:9]
            points_data = map(float, data[0:8])
            #points_data = map(int, points_data)
            #str_data = str(data[8])
            points= [[points_data[0],points_data[1]],\
            [points_data[2],points_data[3]], [points_data[4],points_data[5]],\
            [points_data[6],points_data[7]]]
            quadboxes.append(points)
    if f:
        f.close()
    if len(quadboxes) != 1:
        print('label more than more plate in a image')
    return points#quadboxes

def write_multi_imgs(det_folder, gt_folder ,image_name, org_img, quadboxes, gtboxes):
    for i, quadr in enumerate([0.0, 1.0/32.0, 1.0/16.0]):
        dilated_boxes = dilating(quadboxes, [quadr,quadr,quadr,quadr])
        plate_img = getplateimage(org_img, dilated_boxes.reshape([-1,2]))
        detname = os.path.join(det_folder, '%d_'%i+image_name)
        cv2.imwrite(detname, plate_img)

        dilated_boxes = dilating(gtboxes, [quadr,quadr,quadr,quadr])
        plate_img = getplateimage(org_img, dilated_boxes.reshape([-1,2]))
        gtname = os.path.join(gt_folder, '%d_'%i+image_name)
        cv2.imwrite(gtname, plate_img)

# detecting the test.txt images
def plate_detect(text_detector, root_path, result_prefix):
    base_path = root_path
    img_folder = os.path.join(base_path, 'image')
    missed_folder = os.path.join(base_path, result_prefix+'_missed')
    det_folder = os.path.join(base_path, result_prefix+'_det')
    gt_folder = os.path.join(base_path, result_prefix+'_gt')
    TP = 0
    GT = 0
    FP = 0
    time = Timer()
    if not os.path.isdir(det_folder):
        os.makedirs(det_folder)
    if not os.path.isdir(gt_folder):
        os.makedirs(gt_folder)
    if not os.path.isdir(missed_folder):
        os.makedirs(missed_folder)
    for image_name in os.listdir(img_folder):
        if image_name[-4:] in ['.jpg', '.png', '.JPG', 'JPEG']:
            image_path = os.path.join(img_folder, image_name)
            label_path = image_path[:-4]+'.txt'
            if not os.path.isfile(label_path):
                continue
            #print(image_path)
            detname = os.path.join(det_folder, image_name)
            missedname = os.path.join(missed_folder, image_name)
            print(image_path)
            img = cv2.imread(image_path)
            org_img = img.copy()
            if img is None:
                print('img None\n')
                continue
            start = time.tic()
            regions = text_detector.detect(img)
            print(time.toc())
            quadgt = np.asarray(labelfromtxt(label_path))
            #with open(txtname, 'a') as f:
            missed_flag = 1
            for i in range(len(regions)):
                # show bndbox
                boxes = regions[i]
                #print(boxes[4])
                cv2.rectangle(img, (boxes[0].astype(np.int32), boxes[1].astype(np.int32)), \
                        (boxes[2].astype(np.int32), boxes[3].astype(np.int32)), color=(255, 255, 0), thickness=5)
                # show quadbox
                quadboxes = boxes[10:18]
                dilated_boxes = dilating(quadboxes)
                showrbox(img, dilated_boxes.reshape([-1,2]))
                plate_img = getplateimage(org_img, dilated_boxes.reshape([-1,2]))
                all_img = img_wrapper(img, plate_img, i)
                #bb = [quadboxes[0],quadboxes[1],quadboxes[4],quadboxes[5]]
                #overlap, idx = iou(gtbb, bb)
                overlap = shared_mc(quadgt, quadboxes.reshape([-1,2]))
                if overlap > 0.5:
                    #print(detname)
                    if missed_flag == 1:
                        TP += 1
                    missed_flag = 0
                    #write_multi_imgs(det_folder, gt_folder, image_name, org_img, quadboxes, quadgt.reshape([-1,8])[0])
                    #cv2.imwrite(detname, plate_img)
                else:
                    FP += 1
            cv2.imwrite(detname, all_img)
            if missed_flag:
                cv2.imwrite(missedname, img)
            GT += 1
    print('GT: %d\nrecall: %.5f\nprecision: %.5f\n' %(GT, float(TP)/float(GT), float(TP)/float(TP+FP)))

def plate_detector_test(text_detector, root_path, result_prefix):
    time = Timer()
    for image_name in os.listdir(root_path):
        if image_name[-4:] in ['.jpg', '.png', '.JPG', 'JPEG']:
            image_path = os.path.abspath(os.path.join(root_path, image_name))
            det_path = os.path.abspath('./'+result_prefix)
            detname = os.path.join(det_path, image_name)
            print('1-----------1')
            print(image_path)
            img = cv2.imread(image_path)
            if img is None:
                print('img None\n')
                continue
            org_img = img.copy()
            start = time.tic()
            regions = text_detector.detect(img)
            #print(time.toc())
            #print(len(regions))
            for i in range(len(regions)):
                # show bndbox
                boxes = regions[i]
                print(boxes[4])
                cv2.rectangle(img, (boxes[0].astype(np.int32), boxes[1].astype(np.int32)), \
                        (boxes[2].astype(np.int32), boxes[3].astype(np.int32)), color=(255, 255, 0), thickness=5)
                # show quadbox
                quadboxes = boxes[10:18]
                dilated_boxes = dilating(quadboxes)
                showrbox(img, dilated_boxes.reshape([-1,2]))
                plate_img = getplateimage(org_img, dilated_boxes.reshape([-1,2]))
                all_img = img_wrapper(img, plate_img, i)
            if len(regions)>0:
                cv2.imwrite(detname, all_img)
            print('2-----------2')

if __name__ == '__main__':
    #total_missed = 0
    text_detector = ObjectDetector('mobile',
                                    './output/mobile/VPText_trainval/default/mobile_faster_rcnn_iter_25000.ckpt',
                                    './experiments/cfgs/plate.yml',
                                    num_classes=2,conf_thresh=0.2, nms_thresh=0.5)
    plate_detector_test(text_detector, './test_imgs', 'mobilenet_out')
    '''
    det_dir = '/data/plate/recognize/test'
    for ndir in os.listdir(det_dir):
        det_root = os.path.join(det_dir, ndir)
        if os.path.isdir(det_root):
            print(det_root)
            plate_detect(text_detector, det_root, result_prefix='mobile50000')
    '''

