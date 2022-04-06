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

- Task 1.1 Block Matching code is in `blockMatching.ipynb`

- Task 1.2 visualisation done in `task_1-2.ipynb`, with `.png` results stored in the `task_1-2_results/` folder. Perceiver-IO is ran using `Perceiver - optical_flow.ipynb` on google colab.

- Task 1.3 is in `task1_3.ipynb`

- The main class used along the week project is VehicleDetection, stored in `VehicleDetection.py`. Stores information per detection, the main attributes are the frame number, the confidence value and the 4 corners of the bounding box. It also implements functions like IoU.

- Functions to evaluate (adapted AP from detectron and multiple AP with random shuffling of detections without confidence) detections bounding boxes are located in `eval_utils.py`

- Functions to generate animated GIFs with the bounding boxes are located in `video_utils.py`






### Extra results for Task 1.1
More result examples of the grid search performed can be generated using `plot_3d.py`. The possible arguments are:
  - -d DATA, --data DATA  Relative path to json with grid search data
  - -c COMPENSATION, --compensation COMPENSATION
                        Compensation type: backward - forward
  - -f FUNCTION, --function FUNCTION
                        Distance function: ncc - ssd - sad
  - -s STRIDE, --stride STRIDE
                        Stride value used in grid seatch (data.json contains 1 and 2)
