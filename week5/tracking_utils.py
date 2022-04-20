import motmetrics as mm
import numpy as np
from collections import deque
import pandas as pd
from optical_flow import compute_block_of
import cv2

def create_accumulator():
    # Creates accumulator

    return mm.MOTAccumulator(auto_id=True)

def update_accumulator(acc, gt_object_ids, det_object_ids, IoUs):
    # Updates accumulator with info from current frame
    #   gt_object_ids: List with ground truth object ids in the frame
    #   det_object_ids: List with detection object ids in the frame
    #   IoUs: For every ground truth object id, IoU scores relative to every detection (list of lists)

    frame_id = acc.update(
        gt_object_ids,
        det_object_ids,
        IoUs #inverted (1-IoU version to reward lower values)
    )

    return acc, frame_id

def display_metrics(acc, selected_metrics = ['num_frames', 'precision', 'recall', 'idp', 'idr', 'idf1']):
    # Computes and displays multiple object tracking metrics 

    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=selected_metrics, name='acc')
    print(summary)


def get_box_center(bbox):
    center = (int(bbox[0]) + ((int(bbox[2]) - int(bbox[0])) // 2)), (int(bbox[1]) + ((int(bbox[3]) - int(bbox[1])) // 2)) # X + Width / 2, Y + Height / 2
    return center

# Detection Pre-Processing 

# Remove overlaps in the same frame
def remove_overlaps(detections_pd, tolerance=0.9):
    row_to_remove = []
    for detection in detections_pd.get('detection'):
        length, _ =detections_pd.shape
        for i in range(length):
            IoU = detection.IoU(detections_pd.iloc[i]['detection'])
            if IoU > tolerance and IoU < 1:
                row_to_remove.append(i)
                
    row_to_remove = np.unique(np.array(row_to_remove))
    detections_pd = detections_pd.drop(index=row_to_remove)
        
    return detections_pd


def update_track(detections_pd, next_detections_pd, tolerance=0.5, imgs=None):
    detections_pd['updated'] = False
    detections_pd = detections_pd.reset_index(drop=True)
    
    # Loop each new detection
    for index, next_detection in next_detections_pd.iterrows():
        length, _ = detections_pd.shape
        
        # Find overlaps with max IoU and update if found
        IoUlist = []
        for i in range(length):
            IoU = next_detection['detection'].IoU(detections_pd.iloc[i]['detection'])
            IoUlist.append(IoU)
            
        indexMax = IoUlist.index(max(IoUlist))
            
        if max(IoUlist) > 0.5 and detections_pd.at[indexMax,'updated'] != True:
            detections_pd.at[indexMax,'detection'] = next_detection['detection']
            detections_pd.at[indexMax,'bbox'] = next_detection['bbox']
            detections_pd.at[indexMax,'size'] = next_detection['size']
            detections_pd.at[indexMax,'line'].append(next_detection['line'][0])
            detections_pd.at[indexMax,'updated'] = True
            next_detections_pd.at[index, 'updated'] = True
    
    #if images is not None, use optical flow approach to improve tracking
    if imgs is not None:
        idxs = detections_pd.index[detections_pd['updated'] == False].tolist()
        for ind in idxs:
            box = detections_pd.at[ind, 'bbox']
            block = imgs[1][int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            #Compute optical flow of the not updated tracking bounding box
            of = compute_block_of(block, box, imgs[0], 'backward', 10)
            #update bounding box
            newBox = [box[0]+of[0], box[1]+of[1], box[2]+of[0], box[3]+of[1]]
            detections_pd.at[ind, 'bbox'] = newBox
            detections_pd.at[ind,'updated'] = True
            detections_pd.at[indexMax,'detection'].updateBBox(newBox)
            detections_pd.at[indexMax,'line'].append(get_box_center(newBox))

    # Drop detections no longer exist
    detections_pd = detections_pd[detections_pd['updated'] == True]
                
    # Start tracking new detections
    counter = 0
    if any(next_detections_pd['updated'] == False):
        new_pd = next_detections_pd[next_detections_pd['updated']==False]
        new_pd = new_pd.reset_index(drop=True)
        
        # Generate new track number      
        for i in range(len(new_pd)):
            while counter in detections_pd['track'].tolist():
                counter = counter + 1
            new_pd.at[i, 'track'] = counter
            counter = counter + 1

        # Add new tracks
        detections_pd = pd.concat([detections_pd, new_pd])
            
    detections_pd = detections_pd.reset_index(drop=True)
    return detections_pd


# Detection to DataFrame
def get_detection_dataframe(detections, iclLineAndUpdate = True, firstFrame = False):
    bboxes = []
    bsizes = []
    lines = deque(maxlen=32)
    bdetections = []

    if firstFrame == True:
        tracks = list(range(1, len(detections)+1))
    else:    
        tracks = [0]*len(detections)

    colours = []
    for i in range(len(detections)):
        colours.append(tuple(np.random.choice(range(256), size=3).astype('int')))
        
    updated = [False]*len(detections)
    use_of = [0]*len(detections)
    
    for detection in detections:
        bbox = np.array(detection.getBBox()).astype('int')
        bboxes.append(bbox)
        
        centers = []
        centers.append(get_box_center(bbox))
        lines.append(centers)

        bsize = int(detection.areaOfRec())
        bsizes.append(bsize)

        bdetections.append(detection)

    if iclLineAndUpdate == True: 
        detec = {
            'track': tracks,
            'detection': bdetections,
            'bbox': bboxes,
            'size': bsizes,
            'line': lines,
            'colour': colours,
            'updated': updated,
            'opt_flow': use_of
        }
    else:
        detec = {
            'track': tracks,
            'detection': bdetections,
            'bbox': bboxes,
            'size': bsizes,
            'colour': colours,
        }
    detections_pd = pd.DataFrame(detec)
    detections_pd = detections_pd.sort_values(by=['size'], ascending=False)
    detections_pd = detections_pd.reset_index(drop=True)
    
    return detections_pd

def drawTrackingOnImage(img, bbox, track=0, line=[], colour=(0, 255, 0), showTracking = True):
    b, g, r = colour
    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (int(b), int(g), int(r)), 3)
    if showTracking:
        img = cv2.putText(img, str(track), (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (int(b), int(g), int(r)), 3)
        for i in range(1, len(line)):
            img = cv2.line(img, line[i - 1], line[i], (int(b), int(g), int(r)), 3)
    return img