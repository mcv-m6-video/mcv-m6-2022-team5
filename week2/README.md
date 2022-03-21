# Week 2

## Datasets

* AICityChallenge

* KITTI: flow 2012

## Tasks

* Task 1: Gaussian modelling.
* Task 2: Recursive Gaussian modelling.
* Task 3: Compare with state-of-the-art.
* Task 4: Color space effect.

## Deliverables


- The main class used along the week project is VehicleDetection, stored in `VehicleDetection.py`. Stores information per detection, the main attributes are the frame number, the confidence value and the 4 corners of the bounding box. It also implements functions like IoU.

- Functions to evaluate (adapted AP from detectron and multiple AP with random shuffling of detections without confidence) detections bounding boxes are located in `eval_utils.py`

- Functions to generate animated GIFs with the bounding boxes are located in `video_utils.py`
