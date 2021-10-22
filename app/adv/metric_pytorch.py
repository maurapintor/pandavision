#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 10/30/2018 2:53 PM $ 
# by : shepeleva $ 
# SVN  $
#

# --- imports -----------------------------------------------------------------

import numpy as np

from tqdm import tqdm
import math

import torch

import torch.nn as nn
import torchvision
from torch.autograd import Variable




#  ██╗      ██████╗ ███████╗███████╗    ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
#  ██║     ██╔═══██╗██╔════╝██╔════╝    ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
#  ██║     ██║   ██║███████╗███████╗    █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
#  ██║     ██║   ██║╚════██║╚════██║    ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
#  ███████╗╚██████╔╝███████║███████║    ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
#  ╚══════╝ ╚═════╝ ╚══════╝╚══════╝    ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
#                                                                                                                 






#  ██████╗ ██╗ ██████╗███████╗    ██╗      ██████╗ ███████╗███████╗
#  ██╔══██╗██║██╔════╝██╔════╝    ██║     ██╔═══██╗██╔════╝██╔════╝
#  ██║  ██║██║██║     █████╗      ██║     ██║   ██║███████╗███████╗
#  ██║  ██║██║██║     ██╔══╝      ██║     ██║   ██║╚════██║╚════██║
#  ██████╔╝██║╚██████╗███████╗    ███████╗╚██████╔╝███████║███████║
#  ╚═════╝ ╚═╝ ╚═════╝╚══════╝    ╚══════╝ ╚═════╝ ╚══════╝╚══════╝
#                                                                  

def dice_loss(y_pred, y_true, **kwargs):
    return DiceLoss()(y_pred, y_true, **kwargs)
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target, **kwargs):
        smooth = kwargs['smooth']
        dif = 0
        for i in range(predict.shape[0]):
            prd = nn.Sigmoid()(predict[i]).contiguous().view(-1)
            tar = target[i].contiguous().view(-1)
            int = (prd * tar).sum()
            dis = ((2. * int + smooth) / (prd.sum() + tar.sum() + smooth))
            dif += 1. - dis
        return dif/predict.shape[0]




#  ███╗   ███╗ █████╗ ██████╗  ██████╗ ██╗███╗   ██╗
#  ████╗ ████║██╔══██╗██╔══██╗██╔════╝ ██║████╗  ██║
#  ██╔████╔██║███████║██████╔╝██║  ███╗██║██╔██╗ ██║
#  ██║╚██╔╝██║██╔══██║██╔══██╗██║   ██║██║██║╚██╗██║
#  ██║ ╚═╝ ██║██║  ██║██║  ██║╚██████╔╝██║██║ ╚████║
#  ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝
#                                                   

def margin(y_pred, y_true, **kwargs):
    true = y_true.max(dim=1)[1]
    return nn.SoftMarginLoss()(y_pred, true, )



#  ███████╗██╗ ██████╗ ███╗   ███╗ ██████╗ ██╗██████╗ 
#  ██╔════╝██║██╔════╝ ████╗ ████║██╔═══██╗██║██╔══██╗
#  ███████╗██║██║  ███╗██╔████╔██║██║   ██║██║██║  ██║
#  ╚════██║██║██║   ██║██║╚██╔╝██║██║   ██║██║██║  ██║
#  ███████║██║╚██████╔╝██║ ╚═╝ ██║╚██████╔╝██║██████╔╝
#  ╚══════╝╚═╝ ╚═════╝ ╚═╝     ╚═╝ ╚═════╝ ╚═╝╚═════╝ 
#                                                     

def sigmoid(y_pred, y_true, **kwargs):
    true = y_true.max(dim=1)[1]
    return nn.BCEWithLogitsLoss()(y_pred, true)



#  ███████╗ ██████╗ ███████╗████████╗███╗   ███╗ █████╗ ██╗  ██╗
#  ██╔════╝██╔═══██╗██╔════╝╚══██╔══╝████╗ ████║██╔══██╗╚██╗██╔╝
#  ███████╗██║   ██║█████╗     ██║   ██╔████╔██║███████║ ╚███╔╝ 
#  ╚════██║██║   ██║██╔══╝     ██║   ██║╚██╔╝██║██╔══██║ ██╔██╗ 
#  ███████║╚██████╔╝██║        ██║   ██║ ╚═╝ ██║██║  ██║██╔╝ ██╗
#  ╚══════╝ ╚═════╝ ╚═╝        ╚═╝   ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
#                                                               

def softmax(y_pred, y_true, **kwargs):
    true = y_true.max(dim=1)[1]
    return nn.CrossEntropyLoss()(y_pred, true)




#  ██╗   ██╗ ██████╗ ██╗      ██████╗     ██╗      ██████╗ ███████╗███████╗
#  ╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗    ██║     ██╔═══██╗██╔════╝██╔════╝
#   ╚████╔╝ ██║   ██║██║     ██║   ██║    ██║     ██║   ██║███████╗███████╗
#    ╚██╔╝  ██║   ██║██║     ██║   ██║    ██║     ██║   ██║╚════██║╚════██║
#     ██║   ╚██████╔╝███████╗╚██████╔╝    ███████╗╚██████╔╝███████║███████║
#     ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝     ╚══════╝ ╚═════╝ ╚══════╝╚══════╝
#                                                                          


def detection_loss(y_pred, y_true, anchors, num_classes, iou_thres, **kwargs):
    return YoloLoss(anchors=anchors, num_classes=num_classes, iou_thres=iou_thres, **kwargs)(y_pred, y_true)
    #return compute_loss(y_pred, y_true, **kwargs)
    
class YoloLoss(nn.Module):
    def __init__(self, anchors=[(1.3221, 1.73145), 
                                (3.19275, 4.00944), 
                                (5.05587, 8.09892), 
                                (9.47112, 4.84053),
                                (11.2364, 10.0071)], 
                                reduction=32,num_classes=20, coord_scale=1.0, noobject_scale=1.0,
                                object_scale=5.0, class_scale=1.0, iou_thres=0.6, **kwargs):
        
        
        super(YoloLoss, self).__init__()
        self.num_anchors = len(anchors)
        self.anchor_step = len(anchors[0])
        self.anchors = torch.Tensor(anchors)
        self.reduction = reduction

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.iou_thres = iou_thres
        self.num_classes = num_classes #kwargs.get('num_classes', num_classes)

    def forward(self, output, target):
        batch_size = output.data.size(0)
        height = output.data.size(2)
        width = output.data.size(3)

        # Get x,y,w,h,conf,cls
        output = output.view(batch_size, self.num_anchors, -1, height * width)
        coord = torch.zeros_like(output[:, :, :4, :])
        coord[:, :, :2, :] = output[:, :, :2, :].sigmoid()
        coord[:, :, 2:4, :] = output[:, :, 2:4, :]
        conf = output[:, :, 4, :].sigmoid()
        cls = output[:, :, 5:, :].contiguous().view(batch_size * self.num_anchors, self.num_classes,
                                                    height * width).transpose(1, 2).contiguous().view(-1, self.num_classes)
        # Create prediction boxes
        pred_boxes = torch.FloatTensor(batch_size * self.num_anchors * height * width, 4)
        lin_x = torch.range(0, width - 1).repeat(height, 1).view(height * width)
        lin_y = torch.range(0, height - 1).repeat(width, 1).t().contiguous().view(height * width)
        anchor_w = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)

        if torch.cuda.is_available():
            pred_boxes = pred_boxes.cuda()
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1)#.div(width)
        pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)#.div(height)
        pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)#.div(width)
        pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)#.div(height)
        pred_boxes = pred_boxes.cpu()

        # Get target values
        coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = self.build_targets(pred_boxes, target, height, width)
        coord_mask = coord_mask.expand_as(tcoord)
        tcls = tcls[cls_mask].view(-1).long()
        cls_mask = cls_mask.view(-1, 1).repeat(1, self.num_classes)

        if torch.cuda.is_available():
            tcoord = tcoord.cuda()
            tconf = tconf.cuda()
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()
            tcls = tcls.cuda()
            cls_mask = cls_mask.cuda()

        conf_mask = conf_mask.sqrt()
        cls = cls[cls_mask].view(-1, self.num_classes)

        # Compute losses
        mse = nn.MSELoss(size_average=False) #Mean Squared Error
        ce = nn.CrossEntropyLoss(size_average=False)
        
        self.loss_coord = self.coord_scale * mse(coord * coord_mask, tcoord * coord_mask) / batch_size # localization loss
        self.loss_conf = mse(conf * conf_mask, tconf * conf_mask) / batch_size   #confidence loss
        self.loss_cls = self.class_scale * 2 * ce(cls, tcls) / batch_size         #classification loss
       
        self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls
        
        return self.loss_tot

    def build_targets(self, pred_boxes, ground_truth, height, width):
        batch_size = len(ground_truth)
        
        #print ("ground_truth")
        #print (ground_truth)

        conf_mask  = torch.ones (batch_size, self.num_anchors, height * width,    requires_grad=False) * self.noobject_scale
        
        coord_mask = torch.zeros(batch_size, self.num_anchors, 1, height * width, requires_grad=False)
        cls_mask   = torch.zeros(batch_size, self.num_anchors, height * width,    requires_grad=False, dtype=torch.bool)
        tcoord     = torch.zeros(batch_size, self.num_anchors, 4, height * width, requires_grad=False)
        tconf      = torch.zeros(batch_size, self.num_anchors, height * width,    requires_grad=False)
        tcls       = torch.zeros(batch_size, self.num_anchors, height * width,    requires_grad=False)

        for b in range(batch_size):
            if len(ground_truth[b]) == 0:
                continue

            # Build up tensors
            cur_pred_boxes = pred_boxes[b * (self.num_anchors * height * width) : (b + 1) * (self.num_anchors * height * width)]
            
            if self.anchor_step == 4:
                anchors = self.anchors.clone()
                anchors[:, :2] = 0
            else:
                anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)
            gt = torch.zeros(len(ground_truth[b]), 4)
            for i, anno in enumerate(ground_truth[b]):
                gt[i, 0] = (anno[0] + anno[2] / 2) / self.reduction
                gt[i, 1] = (anno[1] + anno[3] / 2) / self.reduction
                gt[i, 2] = anno[2] / self.reduction
                gt[i, 3] = anno[3] / self.reduction

            # Set confidence mask of matching detections to 0
            iou_gt_pred = YoloLoss.bbox_ious(gt, cur_pred_boxes)
            mask = (iou_gt_pred > self.iou_thres).sum(0) >= 1
            conf_mask[b][mask.view_as(conf_mask[b])] = 0

            # Find best anchor for each ground truth
            gt_wh = gt.clone()
            gt_wh[:, :2] = 0
            iou_gt_anchors = YoloLoss.bbox_ious(gt_wh, anchors)
            _, best_anchors = iou_gt_anchors.max(1)

            # Set masks and target values for each ground truth
            for i, anno in enumerate(ground_truth[b]):
                gi = min(width - 1, max(0, int(gt[i, 0])))
                gj = min(height - 1, max(0, int(gt[i, 1])))
                best_n = best_anchors[i]
                iou = iou_gt_pred[i][best_n * height * width + gj * width + gi]
                coord_mask[b][best_n][0][gj * width + gi] = 1
                cls_mask[b][best_n][gj * width + gi] = True
                conf_mask[b][best_n][gj * width + gi] = self.object_scale
                tcoord[b][best_n][0][gj * width + gi] = gt[i, 0] - gi
                tcoord[b][best_n][1][gj * width + gi] = gt[i, 1] - gj
                tcoord[b][best_n][2][gj * width + gi] = math.log(max(gt[i, 2], 1.0) / self.anchors[best_n, 0])
                tcoord[b][best_n][3][gj * width + gi] = math.log(max(gt[i, 3], 1.0) / self.anchors[best_n, 1])
                tconf[b][best_n][gj * width + gi] = iou
                tcls[b][best_n][gj * width + gi] = int(anno[4])

        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls

    @staticmethod
    def bbox_ious(boxes1, boxes2):
        b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
        b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
        b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
        b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

        dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
        dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
        intersections = dx * dy

        areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
        areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
        unions = (areas1 + areas2.t()) - intersections

        return intersections / unions






#   █████╗  ██████╗ ██████╗██╗   ██╗██████╗  █████╗  ██████╗██╗   ██╗
#  ██╔══██╗██╔════╝██╔════╝██║   ██║██╔══██╗██╔══██╗██╔════╝╚██╗ ██╔╝
#  ███████║██║     ██║     ██║   ██║██████╔╝███████║██║      ╚████╔╝ 
#  ██╔══██║██║     ██║     ██║   ██║██╔══██╗██╔══██║██║       ╚██╔╝  
#  ██║  ██║╚██████╗╚██████╗╚██████╔╝██║  ██║██║  ██║╚██████╗   ██║   
#  ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   
#                                                                    

# ACCURACY FUNCTIONS






#  ██████╗ ███████╗██████╗  ██████╗███████╗███╗   ██╗████████╗ █████╗  ██████╗ ███████╗
#  ██╔══██╗██╔════╝██╔══██╗██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██╔════╝ ██╔════╝
#  ██████╔╝█████╗  ██████╔╝██║     █████╗  ██╔██╗ ██║   ██║   ███████║██║  ███╗█████╗  
#  ██╔═══╝ ██╔══╝  ██╔══██╗██║     ██╔══╝  ██║╚██╗██║   ██║   ██╔══██║██║   ██║██╔══╝  
#  ██║     ███████╗██║  ██║╚██████╗███████╗██║ ╚████║   ██║   ██║  ██║╚██████╔╝███████╗
#  ╚═╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
#                                                                                

def percentage(y_pred, y_true, **kwargs):
    """
    Computes percentage of correct predictions
    :param y_pred:
    :param y_true:
    :return:
    """
    y_pred_soft = nn.Softmax(dim=1)(y_pred)
    perc = (y_pred_soft.max(dim=1)[1] == y_true.max(dim=1)[1]).sum()
    return perc.float()/y_pred_soft.shape[0]



#  ██╗ ██████╗ ██╗   ██╗                                                               
#  ██║██╔═══██╗██║   ██║                                                               
#  ██║██║   ██║██║   ██║                                                               
#  ██║██║   ██║██║   ██║                                                               
#  ██║╚██████╔╝╚██████╔╝                                                               
#  ╚═╝ ╚═════╝  ╚═════╝                                                                
#      

def IoU(y_pred, y_true, **kwargs):
    y_pred_ = nn.Sigmoid()(y_pred)
    y_pred_ = y_pred_.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)
    intersection = (y_pred_ * y_true).sum()
    score = (intersection) / (y_pred_.sum() + y_true.sum() - intersection)

    return score
    
    
    




#  ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
#  ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
#  ██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
#  ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
#  ██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
#  ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
#                                                                                    

def predictions(logits, image_size=416, conf_threshold=0.5, nms_threshold=0.5,
                    anchors=[(1.3221, 1.73145),
                             (3.19275, 4.00944),
                             (5.05587, 8.09892),
                             (9.47112, 4.84053),
                             (11.2364, 10.0071)]):
    num_anchors = len(anchors)
    anchors = torch.Tensor(anchors)
    if isinstance(logits, Variable):
        logits = logits.data

    if logits.dim() == 3:
        logits.unsqueeze_(0)

    batch = logits.size(0)
    w = logits.size(2)
    h = logits.size(3)

    # Compute xc,yc, w,h, box_score on Tensor
    lin_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h * w)
    lin_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().contiguous().view(h * w)
    anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1)
    anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1)
    if torch.cuda.is_available():
        lin_x = lin_x.cuda()
        lin_y = lin_y.cuda()
        anchor_w = anchor_w.cuda()
        anchor_h = anchor_h.cuda()

    logits = logits.view(batch, num_anchors, -1, h * w)
    logits[:, :, 0, :].sigmoid_().add_(lin_x).div_(w) #xc
    logits[:, :, 1, :].sigmoid_().add_(lin_y).div_(h) #yc
    logits[:, :, 2, :].exp_().mul_(anchor_w).div_(w)  #w
    logits[:, :, 3, :].exp_().mul_(anchor_h).div_(h)  #h
    logits[:, :, 4, :].sigmoid_()                     #box score

    # print("  post_processing conf",logits[:, :, 4, :].max(),logits[:, :, 4, :].sum()/logits[:, :, 4, :].numel())

    with torch.no_grad():
        cls_scores = torch.nn.functional.softmax(logits[:, :, 5:, :], 2)
    cls_max, cls_max_idx = torch.max(cls_scores, 2)
    cls_max_idx = cls_max_idx.float()
    cls_max.mul_(logits[:, :, 4, :])

    score_thresh = cls_max > conf_threshold
    score_thresh_flat = score_thresh.view(-1)

    if score_thresh.sum() == 0:
        predicted_boxes = []
        for i in range(batch):
            predicted_boxes.append(torch.Tensor([]))
    else:
        coords = logits.transpose(2, 3)[..., 0:4]
        coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
        scores = cls_max[score_thresh]
        idx = cls_max_idx[score_thresh]
        detections = torch.cat([coords, scores[:, None], idx[:, None]], dim=1)

        max_det_per_batch = num_anchors * h * w
        slices = [slice(max_det_per_batch * i, max_det_per_batch * (i + 1)) for i in range(batch)]
        det_per_batch = torch.IntTensor([score_thresh_flat[s].int().sum() for s in slices])
        split_idx = torch.cumsum(det_per_batch, dim=0)

        # Group detections per image of batch
        predicted_boxes = []
        start = 0
        for end in split_idx:
            predicted_boxes.append(detections[start: end])
            start = end

    selected_boxes = []
    for boxes in predicted_boxes:
        if boxes.numel() == 0:
            selected_boxes.append(torch.Tensor([]))
        else:

          a = boxes[:, :2]
          b = boxes[:, 2:4]
          bboxes_cpy = torch.cat([a - b / 2, a + b / 2], 1)
          scores = boxes[:, 4]

          # Sort coordinates by descending score
          scores, order = scores.sort(0, descending=True)
          x1, y1, x2, y2 = bboxes_cpy[order].split(1, 1)

          # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
          dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp(min=0)
          dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp(min=0)

          # Compute iou
          intersections = dx * dy
          areas = (x2 - x1) * (y2 - y1)
          unions = (areas + areas.t()) - intersections
          ious = intersections / unions

          ious[ious < nms_threshold] = 0
          # Filter based on iou (and class)
          conflicting = ious.triu(1)

          keep = conflicting.sum(0)
          keep = keep.cpu()
          conflicting = conflicting.cpu()

          keep_len = len(keep) - 1
          for i in range(1, keep_len):
              if keep[i] > 0:
                  keep -= conflicting[i]
          if torch.cuda.is_available():
              keep = keep.cuda()

          keep = (keep == 0)
          selected_boxes.append(boxes[order][keep[:, None].expand_as(boxes)].view(-1, 6).contiguous())

    final_boxes = []
    for boxes in selected_boxes:
        if boxes.dim() < 2:
            final_boxes.append(torch.Tensor([]))
        else:
            boxes[:, 0:3:2] *= image_size
            boxes[:, 0]     -= boxes[:, 2] / 2
            boxes[:, 2]      = boxes[:, 0] + boxes[:, 2]
            boxes[:, 1:4:2] *= image_size
            boxes[:, 1]     -= boxes[:, 3] / 2
            boxes[:, 3]      = boxes[:, 1] + boxes[:, 3]

            final_boxes.append(torch.tensor([[box[0].item(),  # x1
                                              box[1].item(),  # y1
                                              box[2].item(),  # x2
                                              box[3].item(),  # y2
                                              box[4].item(),  # conf
                                          int(box[5].item())  # class
                                          ] for box in boxes]))
    
    outputs = []
    for i in range(len(final_boxes)):
      if len(final_boxes[i])>0:
        
        outputs.append(torch.stack([final_boxes[i][:, 0],                           # x
                            final_boxes[i][:, 1],                           # y
                            final_boxes[i][:, 2] - final_boxes[i][:, 0],    # w
                            final_boxes[i][:, 3] - final_boxes[i][:, 1],    # h
                            final_boxes[i][:, 4],                           # conf
                            final_boxes[i][:, 5]],                          # class
                            dim=1))
      else:
        outputs.append(final_boxes[i])
    
    
    return outputs



#  ██╗███╗   ███╗ █████╗  ██████╗ ███████╗    ██╗      ██████╗  ██████╗  ██████╗ ███████╗██████╗ 
#  ██║████╗ ████║██╔══██╗██╔════╝ ██╔════╝    ██║     ██╔═══██╗██╔════╝ ██╔════╝ ██╔════╝██╔══██╗
#  ██║██╔████╔██║███████║██║  ███╗█████╗      ██║     ██║   ██║██║  ███╗██║  ███╗█████╗  ██████╔╝
#  ██║██║╚██╔╝██║██╔══██║██║   ██║██╔══╝      ██║     ██║   ██║██║   ██║██║   ██║██╔══╝  ██╔══██╗
#  ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝███████╗    ███████╗╚██████╔╝╚██████╔╝╚██████╔╝███████╗██║  ██║
#  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚══════╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝
#                                                                                                



#VOC order
#{"airplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
#"cat", "chair", "cow", "dining table", "dog", "horse", "motorcycle", "person", "potted plant", "sheep", "couch", "train", "tv"}

#COCO order
#{"person","bicycle","car","motorcycle","airplane","bus","train","boat","bird","cat","dog","horse",
#"sheep","cow","bottle","chair","couch","potted plant","dining table","tv"}

class Image_logger:
    def __init__(self, tbx_writer=None, tag='image', img_size=416):
        self.done_for_epoch = False;
        self.tbx_writer = tbx_writer
        self.tag        = tag
        self.img_size   = img_size
        self.lcolors =  [[255, 0,  128],
                         [0,   255, 255],
                         [128, 128, 0],
                         [0,   128, 128],
                         [128, 0,   255],
                         [0,   0,   0],
                         [0,   255, 128],
                         [0,   255, 0],
                         [128, 255, 255],
                         [0,   0,   128],
                         [128, 0,   0],
                         [128, 0,   128],
                         [0,   128, 0],
                         [128, 255, 0],
                         [128, 128, 255],
                         [0,   128, 255],
                         [128, 128, 128],
                         [0,   0,   255],
                         [128, 255, 128],
                         [255, 0,   0]
                         ]
 
        self.lcolors = [[x / 255 for x in c] for c in self.lcolors]


    def clear(self):
        self.done_for_epoch = False
        

        
    def draw_results(self, predictions, tinput, epoch) :
        
        if (not self.done_for_epoch) and (len(predictions)) > 0:
        
        
        #    print("######### predictions ", self.tag +  str(len(predictions)))
            rand_idx = np.arange(len(predictions))
            np.random.shuffle(rand_idx)
            for idx in np.nditer(rand_idx[:2]):
                pred = predictions[idx]
                if len(pred):
                    timage = tinput[idx,:,:,:].clone()
                    Image_logger.draw_bb_list(timage, pred, self.img_size, self.lcolors)
               #     self.tbx_writer.add_image(self.tag + str(idx),timage[torch.LongTensor([2,1,0])],epoch)
                    self.tbx_writer.add_image(self.tag + str(idx),timage,epoch)
                    self.done_for_epoch = True
    @staticmethod
    def draw_bb( image, bbi, img_size, lcolors):
        
        bb = [max(0, min(img_size[0] - 1, bbi[0])),
              max(0, min(img_size[1] - 1, bbi[1])),
              min(img_size[0] - 1, max(0, bbi[2]+bbi[0])),
              min(img_size[1] - 1, max(0, bbi[3]+bbi[1]))]
              
        bb = [int(i) for i in bb]
        #print("image",image.shape)
        #print("image bb",bb)
        #classl = 0
        classl = int(bbi[5])
        if bb[0]<bb[2] and bb[1]<bb[3]:
          for y in range(bb[1],bb[3]):
              image[0][y][bb[0]] = lcolors[classl][0]
              image[1][y][bb[0]] = lcolors[classl][1]
              image[2][y][bb[0]] = lcolors[classl][2]
              image[0][y][bb[2]] = lcolors[classl][0]
              image[1][y][bb[2]] = lcolors[classl][1]
              image[2][y][bb[2]] = lcolors[classl][2]

          for x in range(bb[0],bb[2]):
              image[0][bb[1]][x] = lcolors[classl][0]
              image[1][bb[1]][x] = lcolors[classl][1]
              image[2][bb[1]][x] = lcolors[classl][2]
              image[0][bb[3]][x] = lcolors[classl][0]
              image[1][bb[3]][x] = lcolors[classl][1]
              image[2][bb[3]][x] = lcolors[classl][2]

    @staticmethod
    def draw_bb_list( image, bbi_list,img_size, lcolors):
        for bb in bbi_list[:20]:
            Image_logger.draw_bb(image,bb,img_size, lcolors)

