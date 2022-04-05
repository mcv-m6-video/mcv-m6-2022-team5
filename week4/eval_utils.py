from matplotlib import pyplot as plt
from tracking_utils import *
from tqdm import tqdm
import numpy as np
import random

#############################################################################################################
# Code extracted from detectron2
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/pascal_voc_evaluation.py
#############################################################################################################

def tracking_acc(frame, gt_detect, detections, acc, max_iou=1):
    sframe = str(frame)

    gt_detections_pd = get_detection_dataframe(gt_detect[sframe])
    det_detections_pd = detections[frame]
    
    gt_object_ids = []
    gt_objects_boxes = []
    for track in gt_detections_pd['track']:
        gt_object_ids.append(track)
        x = gt_detections_pd[gt_detections_pd['track'] == track].detection.item().xtl
        y = gt_detections_pd[gt_detections_pd['track'] == track].detection.item().ytl
        w = gt_detections_pd[gt_detections_pd['track'] == track].detection.item().w
        h = gt_detections_pd[gt_detections_pd['track'] == track].detection.item().h
        gt_objects_boxes.append([x, y, w, h])

    det_object_ids = []
    det_objects_boxes = []
    for track in det_detections_pd['track']:
        det_object_ids.append(track)
        x = det_detections_pd[det_detections_pd['track'] == track].detection.item().xtl
        y = det_detections_pd[det_detections_pd['track'] == track].detection.item().ytl
        w = det_detections_pd[det_detections_pd['track'] == track].detection.item().w
        h = det_detections_pd[det_detections_pd['track'] == track].detection.item().h
        det_objects_boxes.append([x, y, w, h])

    distances = mm.distances.iou_matrix(gt_objects_boxes, det_objects_boxes, max_iou)

    acc, frame_id = update_accumulator(acc, gt_object_ids, det_object_ids, distances)

    return acc, frame_id

def voc_ap(rec, prec, use_07_metric=True):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(gt, detections, ovthresh=0.5, use_conf=False):
    """rec, prec, ap, tp_detections = voc_eval(gt,
                                detections,
                                [ovthresh],
                                [use_conf])
    Top level function that does the PASCAL VOC evaluation.
    gt: ground truth annotations
    detections: estiamted detections.
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_conf]: Whether to use confidence or not
        (default False)
    """
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    image_ids, confidence, BB, all_detect = [], [], [], []
    for frame, gt_ds in gt.items():
        #GT data
        bbox = np.array([gt_d.getBBox() for gt_d in gt_ds])
        det = [False] * len(gt_ds)
        npos = npos + len(gt_ds)
        class_recs[frame] = {"bbox": bbox, "det": det}

        if frame in detections:
            #Deected data
            detected = detections[frame]
            all_detect = [*all_detect, *detected]

            #Frame of boxes
            idsAux = [frame] * len(detected)
            image_ids = [*image_ids, *idsAux]

            #Computed confidence
            confAux = [d.conf for d in detected]
            confidence = [*confidence, *confAux]

            #Detected bounding boxes
            BBAux = [d.getBBox() for d in detected]
            BB = [*BB, *BBAux]

    confidence = np.array(confidence)
    BB = np.array(BB).reshape(-1, 4)

    # sort by confidence
    if use_conf:
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        all_detect = [all_detect[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    tpDetections = {}
    IoU_tp = {}
    IoU = {}
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if not R["det"][jmax]:
            if image_ids[d] not in IoU:
                IoU[image_ids[d]] = []
            IoU[image_ids[d]].append(ovmax)
        if ovmax > ovthresh:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = 1
                if image_ids[d] not in tpDetections:
                    tpDetections[image_ids[d]] = []
                    IoU_tp[image_ids[d]] = []
                tpDetections[image_ids[d]].append(all_detect[d])
                IoU_tp[image_ids[d]].append(ovmax)
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return rec, prec, ap, tpDetections, IoU_tp, IoU


def video_IoU(gt, detections):
    mIoUPerFrame = {}
    count = []

    # Iterate all frames
    for frame_num, boxes in gt.items():
        # Get data from frame number: frame_num
        if frame_num in detections:
            frames_data = detections[frame_num]
            scores = []
            # Compare GT VehicleDetection to the Predicted
            for frame_gt in boxes:
                box_iou = []
                for frame_data in frames_data:
                    # Double for because some frames_data contains +1 VehicleDetection
                    iou = frame_gt.IoU(frame_data)
                    box_iou.append(iou)
                scores.append(np.max(box_iou))
            mIoUPerFrame[frame_num] = np.mean(scores)
            count.append(mIoUPerFrame[frame_num])

    mIoU = np.mean(count)
    return mIoU, mIoUPerFrame


def randomizeFrameBoxes(frames):
    for frame, ds in frames.items():
        random.shuffle(ds)
        frames[frame] = ds
    
    return frames

def ap_wo_conf(gt, detections, N=10 ,ovthresh=0.5):
    recs, precs, aps = [], [], []
    for i in range(N):
        rec, prec, ap, tpDetections, IoU_tp, IoU = voc_eval(gt, randomizeFrameBoxes(detections), ovthresh)
        recs.append(rec)
        precs.append(prec)
        aps.append(ap)

    return recs, precs, aps

def plot_prec_recall_curve(prec, rec, title='Precision-Recall curve', xAxis='Recall', yAxis='Precision'):
    # plotting the points
    plt.plot(rec, prec)
    
    # naming the x axis
    plt.xlabel(xAxis)
    # naming the y axis
    plt.ylabel(yAxis)
    
    # giving a title to my graph
    plt.title(title)
    
    # function to show the plot
    plt.show()

def plot_multiple_prec_recall_curves(precs, recs, labels, title='Precision-Recall curve', xAxis='Recall', yAxis='Precision'):
    for ind, rec in enumerate(recs):
        # plotting the points
        plt.plot(rec, precs[ind], label=labels[ind])
    
    # naming the x axis
    plt.xlabel(xAxis)
    # naming the y axis
    plt.ylabel(yAxis)
    
    # giving a title to my graph
    plt.title(title)
    
    plt.legend()
    
    # function to show the plot
    plt.show()