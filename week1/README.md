# Week 1 

## Datasets

* AICityChallenge

* KITTI: flow 2012

## Tasks

* Task 1: Detection metrics.
* Task 2: Detection metrics. Temporal analysis.
* Task 3: Optical flow evaluation metrics.
* Task 4: Visual representation optical flow.

## Deliverables
All tasks are delivered in jupyter notebooks with their corresponding name.

- The main class used along the week project is VehicleDetection, stored in `VehicleDetection.py`. Stores information per detection, the main attributes are the frame number, the confidence value and the 4 corners of the bounding box. It also implements functions like IoU.

- Functions to evaluate (adapted AP from detectron and multiple AP with random shuffling of detections without confidence) detections bounding boxes are located in `eval_utils.py`

- Functions to generate animated GIFs with the bounding boxes are located in `video_utils.py`

- Functions used to read and visuialise optical flow in both task 3 and 4 are located in `optical_flow.py`.