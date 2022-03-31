# Week 3

## Datasets

* AICityChallenge

## Tasks

* Task 1: Object detection
    * Task 1.1: Off-the-shelf
    * Task 1.2: Fine-tune to your data
    * Task 1.3: K-Fold Cross-validation

* Task 2: Object tracking
    * Task 2.1: Tracking by Overlap
    * Task 2.2: Tracking with a Kalman Filter
    * Task 2.3: IDF1 score

## Deliverables

- The first three task are in the correspondent python files that can be identified by their name. To run them you can either use the default set of parameters that works on group05 main folder in the CVC folder or edit them. Use `python fileName -h` to get the help of the arguments.

- For all the content of task 2, jupyter notebooks has been developed.

- The main class used along the week project is VehicleDetection, stored in `VehicleDetection.py`. Stores information per detection, the main attributes are the frame number, the confidence value and the 4 corners of the bounding box. It also implements functions like IoU.

- Functions to evaluate (adapted AP from detectron and multiple AP with random shuffling of detections without confidence) detections bounding boxes are located in `eval_utils.py`

- Functions to generate animated GIFs with the bounding boxes are located in `video_utils.py`

