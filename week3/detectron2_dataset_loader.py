import cv2
from detectron2.structures import BoxMode


def get_AICity_dicts(gt_detected, frames, base_path):
    """
    Registers the KITTI-MOTS dataset to detectron2
    """
    
    dataset_dicts = []
    for file in frames:
        frame = file[4:-4] #format is vdo_<number>.png 
        record = {}
        filename = base_path + file
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = frame
        record["height"] = height
        record["width"] = width
    
        objs = []
        boxes = gt_detected[str(frame)]
        for elems in boxes:
            obj = {
                "bbox": elems.getBBox(),
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0, # Only cars, maybe could be more generic
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
