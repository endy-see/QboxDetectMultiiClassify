# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform, rbox_transform, quadbox_transform
from utils.cython_bbox import bbox_overlaps


def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, gt_rboxes, gt_quadboxes, _num_classes, recogn_labels):
    """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois
    all_scores = rpn_scores

    # Include ground-truth boxes in the set of candidate rois
    if cfg.TRAIN.USE_GT:
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        # rzeros = np.zeros((gt_rboxes.shape[0], 1), dtype=gt_rboxes.dtype)
        # all_rois = np.vstack(
        #   (all_rois, np.hstack((rzeros, gt_rboxes[:, :-1])))
        # )
        # not sure if it a wise appending, but anyway i am not using it
        all_scores = np.vstack((all_scores, zeros))
        # all_scores = np.vstack((all_scores, rzeros))

    num_images = 1
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, roi_scores, bbox_targets, rbox_targets, quadbox_targets, \
    bbox_inside_weights, rbox_inside_weights, quadbox_inside_weights, recogn_labels \
     = _sample_rois(
        all_rois, all_scores, gt_boxes, gt_rboxes, gt_quadboxes, fg_rois_per_image,
        rois_per_image, _num_classes, recogn_labels)

    rois = rois.reshape(-1, 5)
    roi_scores = roi_scores.reshape(-1)
    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
    rbox_targets = rbox_targets.reshape(-1, _num_classes * 5)
    quadbox_targets = quadbox_targets.reshape(-1, _num_classes * 8)

    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
    recogn_labels = recogn_labels.reshape(-1, 1)

    rbox_inside_weights = rbox_inside_weights.reshape(-1, _num_classes * 5)
    rbox_outside_weights = np.array(rbox_inside_weights > 0).astype(np.float32)

    quadbox_inside_weights = quadbox_inside_weights.reshape(-1, _num_classes * 8)
    quadbox_outside_weights = np.array(quadbox_inside_weights > 0).astype(np.float32)
    return rois, roi_scores, labels, bbox_targets, rbox_targets, quadbox_targets, \
           bbox_inside_weights, bbox_outside_weights, \
           rbox_inside_weights, rbox_outside_weights, \
           quadbox_inside_weights, quadbox_outside_weights, \
           recogn_labels


# add rbox_regression_labels
def _get_bbox_regression_labels(bbox_target_data, rbox_target_data, quadbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      rbox_target (ndarray): N x 4K blob of regression targets for rboxes
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)

    rbox_targets = np.zeros((clss.size, 5 * num_classes), dtype=np.float32)
    rbox_inside_weights = np.zeros(rbox_targets.shape, dtype=np.float32)

    quadbox_targets = np.zeros((clss.size, 8 * num_classes), dtype=np.float32)
    quadbox_inside_weights = np.zeros(quadbox_targets.shape, dtype=np.float32)

    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS

        rstart = int(5 * cls)
        rend = rstart + 5
        rbox_targets[ind, rstart:rend] = rbox_target_data[ind, 1:]
        rbox_inside_weights[ind, rstart:rend] = cfg.TRAIN.RBOX_INSIDE_WEIGHTS

        quadstart = int(8 * cls)
        quadend = quadstart + 8
        quadbox_targets[ind, quadstart:quadend] = quadbox_target_data[ind, 1:]
        quadbox_inside_weights[ind, quadstart:quadend] = cfg.TRAIN.QUADBOX_INSIDE_WEIGHTS
    return bbox_targets, rbox_targets, quadbox_targets, bbox_inside_weights, rbox_inside_weights, quadbox_inside_weights


# add rtargets
def _compute_targets(ex_rois, gt_rois, gt_rrois, gt_quadrois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4
    assert gt_rrois.shape[1] == 5
    assert gt_quadrois.shape[1] == 8

    targets = bbox_transform(ex_rois, gt_rois)
    rtargets = rbox_transform(ex_rois, gt_rrois)
    quadtargets = quadbox_transform(ex_rois, gt_quadrois)

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                   / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))

    if cfg.TRAIN.RBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        rtargets = ((rtargets - np.array(cfg.TRAIN.RBOX_NORMALIZE_MEANS))
                    / np.array(cfg.TRAIN.RBOX_NORMALIZE_STDS))

    if cfg.TRAIN.QUADBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        quadtargets = ((quadtargets - np.array(cfg.TRAIN.QUADBOX_NORMALIZE_MEANS))
                       / np.array(cfg.TRAIN.QUADBOX_NORMALIZE_STDS))

    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False), \
           np.hstack(
               (labels[:, np.newaxis], rtargets)).astype(np.float32, copy=False), \
           np.hstack(
               (labels[:, np.newaxis], quadtargets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, all_scores, gt_boxes, gt_rboxes, gt_quadboxes, fg_rois_per_image, rois_per_image,
                 num_classes, recogn_labels):
    """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]
    recogn_labels = recogn_labels[gt_assignment,0]
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Small modification to the original version where we ensure a fixed number of regions are sampled
    if fg_inds.size > 0 and bg_inds.size > 0:
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        to_replace = bg_inds.size < bg_rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
    elif fg_inds.size > 0:
        to_replace = fg_inds.size < rois_per_image
        fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = rois_per_image
    elif bg_inds.size > 0:
        to_replace = bg_inds.size < rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = 0
    else:
        import pdb
        pdb.set_trace()
    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    keep_inds_recogn = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    recogn_labels = recogn_labels[keep_inds_recogn]
    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_image):] = 0
    recogn_labels[int(fg_rois_per_image):] = 0
    rois = all_rois[keep_inds]
    roi_scores = all_scores[keep_inds]
    bbox_target_data, rbox_target_data, quadbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], \
        gt_rboxes[gt_assignment[keep_inds], :5], \
        gt_quadboxes[gt_assignment[keep_inds], :8], \
        labels)
    bbox_targets, rbox_targets, quadbox_targets, bbox_inside_weights, rbox_inside_weights, quadbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, rbox_target_data, quadbox_target_data, num_classes)
    return labels, rois, roi_scores, bbox_targets, rbox_targets, \
           quadbox_targets, bbox_inside_weights, rbox_inside_weights, quadbox_inside_weights, recogn_labels
