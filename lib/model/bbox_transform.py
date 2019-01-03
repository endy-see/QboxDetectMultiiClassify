# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math


def quadbox_transform(ex_rois, gt_quadrois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_quadx1 = gt_quadrois[:, 0]
    gt_quady1 = gt_quadrois[:, 1]
    gt_quadx2 = gt_quadrois[:, 2]
    gt_quady2 = gt_quadrois[:, 3]
    gt_quadx3 = gt_quadrois[:, 4]
    gt_quady3 = gt_quadrois[:, 5]
    gt_quadx4 = gt_quadrois[:, 6]
    gt_quady4 = gt_quadrois[:, 7]

    # try new regression target: minus 1/2 the means of the original rtargets is about 1/2
    quadtargets_dx1 = (gt_quadx1 - ex_ctr_x) / ex_widths
    quadtargets_dy1 = (gt_quady1 - ex_ctr_y) / ex_heights
    quadtargets_dx2 = (gt_quadx2 - ex_ctr_x) / ex_widths
    quadtargets_dy2 = (gt_quady2 - ex_ctr_y) / ex_heights
    quadtargets_dx3 = (gt_quadx3 - ex_ctr_x) / ex_widths
    quadtargets_dy3 = (gt_quady3 - ex_ctr_y) / ex_heights
    quadtargets_dx4 = (gt_quadx4 - ex_ctr_x) / ex_widths
    quadtargets_dy4 = (gt_quady4 - ex_ctr_y) / ex_heights

    quadtargets = np.vstack(
        (quadtargets_dx1, quadtargets_dy1, quadtargets_dx2, quadtargets_dy2, \
         quadtargets_dx3, quadtargets_dy3, quadtargets_dx4, quadtargets_dy4,)).transpose()
    return quadtargets


def quadbox_transform_inv(boxes, quaddeltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, quaddeltas.shape[1]), dtype=quaddeltas.dtype)

    boxes = boxes.astype(quaddeltas.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dquadx1 = quaddeltas[:, 0::8]
    dquady1 = quaddeltas[:, 1::8]
    dquadx2 = quaddeltas[:, 2::8]
    dquady2 = quaddeltas[:, 3::8]
    dquadx3 = quaddeltas[:, 4::8]
    dquady3 = quaddeltas[:, 5::8]
    dquadx4 = quaddeltas[:, 6::8]
    dquady4 = quaddeltas[:, 7::8]

    pred_quadx1 = dquadx1 * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_quady1 = dquady1 * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_quadx2 = dquadx2 * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_quady2 = dquady2 * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_quadx3 = dquadx3 * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_quady3 = dquady3 * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_quadx4 = dquadx4 * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_quady4 = dquady4 * heights[:, np.newaxis] + ctr_y[:, np.newaxis]

    pred_quadboxes = np.zeros(quaddeltas.shape, dtype=quaddeltas.dtype)
    pred_quadboxes[:, 0::8] = pred_quadx1
    pred_quadboxes[:, 1::8] = pred_quady1
    pred_quadboxes[:, 2::8] = pred_quadx2
    pred_quadboxes[:, 3::8] = pred_quady2
    pred_quadboxes[:, 4::8] = pred_quadx3
    pred_quadboxes[:, 5::8] = pred_quady3
    pred_quadboxes[:, 6::8] = pred_quadx4
    pred_quadboxes[:, 7::8] = pred_quady4

    return pred_quadboxes


def rbox_transform(ex_rois, gt_rrois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_rx1 = gt_rrois[:, 0]
    gt_ry1 = gt_rrois[:, 1]
    gt_rx2 = gt_rrois[:, 2]
    gt_ry2 = gt_rrois[:, 3]
    gt_rh = gt_rrois[:, 4]
    # try new regression target: minus 1/2 the means of the original rtargets is about 1/2
    rtargets_drx1 = (gt_rx1 - ex_ctr_x) / ex_widths
    rtargets_dry1 = (gt_ry1 - ex_ctr_y) / ex_heights
    rtargets_drx2 = (gt_rx2 - ex_ctr_x) / ex_widths
    rtargets_dry2 = (gt_ry2 - ex_ctr_y) / ex_heights
    rtargets_drh = np.log(gt_rh / ex_heights)

    rtargets = np.vstack(
        (rtargets_drx1, rtargets_dry1, rtargets_drx2, rtargets_dry2, rtargets_drh)).transpose()
    return rtargets


def rbox_transform_inv(boxes, rdeltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, rdeltas.shape[1]), dtype=rdeltas.dtype)

    boxes = boxes.astype(rdeltas.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    drx1 = rdeltas[:, 0::5]
    dry1 = rdeltas[:, 1::5]
    drx2 = rdeltas[:, 2::5]
    dry2 = rdeltas[:, 3::5]
    drh = rdeltas[:, 4::5]

    pred_rx1 = drx1 * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ry1 = dry1 * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_rx2 = drx2 * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ry2 = dry2 * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_rh = np.exp(drh) * heights[:, np.newaxis]

    pred_rboxes = np.zeros(rdeltas.shape, dtype=rdeltas.dtype)
    # rx1
    pred_rboxes[:, 0::5] = pred_rx1
    # ry1
    pred_rboxes[:, 1::5] = pred_ry1
    # rx2
    pred_rboxes[:, 2::5] = pred_rx2
    # ry2
    pred_rboxes[:, 3::5] = pred_ry2
    # rh
    pred_rboxes[:, 4::5] = pred_rh

    return pred_rboxes


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights

    gt_widths = np.maximum(gt_widths, 1)
    ex_widths = np.maximum(ex_widths, 1)
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
  Clip boxes to image boundaries.
  """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
