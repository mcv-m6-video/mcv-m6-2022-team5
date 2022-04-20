import cv2
import os
import pickle as pkl
from detectron2.structures import BoxMode


def get_AICity_dicts(gt_detected, frames, base_path, div):
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
        if str(idx*div) in gt_detected.keys():
            image_id = idx + last_id
            record = {}
            height, width = frame.shape[:2]
            filename = base_path + f"vdo_{idx*div}.png"
            if not os.path.exists(filename):
                cv2.imwrite(filename, frame)
            record["file_name"] = filename
            record["image_id"] = image_id
            record["height"] = height
            record["width"] = width

            objs = []
            boxes = gt_detected[str(idx*div)]
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

def get_AICity_dicts_big(seqs):
    result = []
    for seq in seqs:
        result += get_AICity_dicts(**seq)
    return result