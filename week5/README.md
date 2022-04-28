# Week 5

## Datasets

* AICityChallenge

* KITTI: flow 2012

## Tasks

* T2 Multi-target single-camera (MTSC) tracking
    * T1.1 Evaluate MTSC on SEQ 3
    * T1.2 Evaluate MTSC on SEQ 1 & SEQ 4
* T2 Multi-target multi-camera (MTMC) tracking
    * T2.1 Evaluate MTMC on SEQ 3
    * T2.2 Evaluate MTMC on SEQ 1 & SEQ 4


## Deliverables

- Notebook files with their respective tasks and an example of execution for revision purposes.

- The main class used along the week project is VehicleDetection, stored in `VehicleDetection.py`. Stores information per detection, the main attributes are the frame number, the confidence value and the 4 corners of the bounding box. It also implements functions like IoU.

- Functions to evaluate (adapted AP from detectron and multiple AP with random shuffling of detections without confidence) detections bounding boxes are located in `eval_utils.py`

- Functions to generate animated GIFs with the bounding boxes are located in `video_utils.py`

- SORT Kalman filter method implemented in `sort.py`
