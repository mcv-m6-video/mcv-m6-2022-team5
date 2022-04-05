import cv2
import os
import pickle as pkl
from detectron2.structures import BoxMode


def get_AICity_dicts(gt_detected, frames, base_path):
    """
    Registers the KITTI-MOTS dataset to detectron2
    """
    
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    last_id = 0
    image_id = 0

    if os.path.exists("last_id.pkl"):
        last_id = pkl.load(open("last_id.pkl", "rb"))[0]


    dataset_dicts = []
    for idx, frame in enumerate(frames):
        if idx*2 in gt_detected.keys():
            image_id = idx + last_id
            record = {}
            height, width = frame.shape[:2]
            filename = base_path + f"vdo_{idx*2}.png"
            cv2.imwrite(filename, frame)
            record["file_name"] = filename
            record["image_id"] = image_id
            record["height"] = height
            record["width"] = width

            objs = []
            boxes = gt_detected[str(idx*2)]
            for elems in boxes:
                obj = {
                    "bbox": elems.getBBox(),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0, # Only cars, maybe could be more generic
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    
    pkl.dump([image_id],open("last_id.pkl", "wb"))
    return dataset_dicts
