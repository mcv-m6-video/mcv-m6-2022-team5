# Week 4

## Datasets

* AICityChallenge

* KITTI: flow 2012

## Tasks

* T1 Optical Flow
    * T1.1 Optical Flow with Block Matching
    * T1.2 Off-the-shelf Optical Flow
    * T1.3 Object Tracking with Optical Flow
* T2 Multi-target single-camera (MTSC) tracking
    * T2.1 Evaluate MTSC on SEQ 3
    * T2.2 Evaluate MTSC on SEQ 1 & SEQ 4


## Deliverables


- The main class used along the week project is VehicleDetection, stored in `VehicleDetection.py`. Stores information per detection, the main attributes are the frame number, the confidence value and the 4 corners of the bounding box. It also implements functions like IoU.

- Functions to evaluate (adapted AP from detectron and multiple AP with random shuffling of detections without confidence) detections bounding boxes are located in `eval_utils.py`

- Functions to generate animated GIFs with the bounding boxes are located in `video_utils.py`

- Task 3 uses the [OpenCV.BackgroundSubstractor](https://docs.opencv.org/3.4/d7/df6/classcv_1_1BackgroundSubtractor.html) algorithms and [BGSLibrary](https://github.com/andrewssobral/bgslibrary) by Andrew Sobral. The implementations are contained with in `task3.ipynb` and `task3_BGSLibrary.py`, respectively.

### Extra results for Task 1.1
More result examples of the grid search performed can be generated using `plot_3d.py`. The possible arguments are:
  - -d DATA, --data DATA  Relative path to json with grid search data
  - -c COMPENSATION, --compensation COMPENSATION
                        Compensation type: backward - forward
  - -f FUNCTION, --function FUNCTION
                        Distance function: ncc - ssd - sad
  - -s STRIDE, --stride STRIDE
                        Stride value used in grid seatch (data.json contains 1 and 2)
