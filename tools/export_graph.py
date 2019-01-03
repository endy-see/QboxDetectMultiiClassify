#!/usr/bin/env python

"""
export the tf-faster-rcnn model's graph text file

python export_graph.py EXPORT mobile /Users/lairf/AI-projects/tf-faster-rcnn/output/mobile_0.25/mobile_faster_rcnn_iter_70000.ckpt \
    /Users/lairf/AI-projects/tf-faster-rcnn/experiments/cfgs/text.yml \
    4 128 /Users/lairf/AI-projects/tf-faster-rcnn/output/ mobile_faster_rcnn.pbtxt
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg, cfg_from_file
import tensorflow as tf
import os, sys
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1


def export(mode, net_name, model_path, config_file, num_classes = 2, feature_channel=128, export_path='./', model_name='faster_rcnn.pbtxt'):  
    if not os.path.isfile(model_path + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                    'our server and place them properly?').format(model_path + '.meta'))
    
    # load network configuration
    cfg_from_file(config_file)
    # pprint.pprint(cfg)

    # load network
    if net_name == 'vgg16':
        net = vgg16(batch_size=1)
    elif net_name == 'res50':
        net = resnetv1(batch_size=1, num_layers=50)
    elif net_name == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    elif net_name == 'res152':
        net = resnetv1(batch_size=1, num_layers=152)
    elif net_name == 'mobile':
        net = mobilenetv1(batch_size=1)
    else:
        raise NotImplementedError

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    # tfconfig=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    # init session
    sess = tf.Session(config=tfconfig)
    with sess.as_default(), tf.device('/cpu:0'):
        net.create_architecture(sess, mode, num_classes, tag='default',
                        anchor_scales=cfg.ANCHOR_SCALES,
                        anchor_ratios=cfg.ANCHOR_RATIOS,
                        feature_channel=feature_channel)

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print('restored from ckpt: {}'.format(model_path))
        tf.train.write_graph(sess.graph_def, export_path, model_name)
        print("exported graph to: {}{}".format(export_path, model_name))

if __name__ == '__main__':
    # load text line recongize model
    if len(sys.argv) < 9:
        print("please provide mode net_name, model_path, config_file, num_classes, feature_channel, export_path, model_name")
    else:
        mode = sys.argv[1]
        net_name = sys.argv[2]
        model_path = sys.argv[3]
        config_file = sys.argv[4]
        num_classes = int(sys.argv[5])
        feature_channel = int(sys.argv[6])
        export_path = sys.argv[7]
        model_name = sys.argv[8]
        export(mode, net_name, model_path, config_file, num_classes, feature_channel, export_path, model_name)