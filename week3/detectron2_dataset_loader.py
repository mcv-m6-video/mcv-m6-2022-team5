import cv2
from detectron2.structures import BoxMode


def get_KITTIMOTS_dicts(gt_detected, frames, format_path):
    """
    Registers the KITTI-MOTS dataset to detectron2
    """
    
    dataset_dicts = []
    for frame in frames:
        record = {}
        
        height, width = cv2.imread(format_path.format(frame)).shape[:2]
        
        record["file_name"] = format_path.format(str(frame))
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
