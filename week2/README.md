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

- Task 3 uses the [OpenCV.BackgroundSubstractor](https://docs.opencv.org/3.4/d7/df6/classcv_1_1BackgroundSubtractor.html) algorithms and [BGSLibrary](https://github.com/andrewssobral/bgslibrary) by Andrew Sobral. The implementations are contained with in `task3.ipynb` and `task3_BGSLibrary.py`, respectively.

### Extra results for Task3
The result presented here used a slightly aggressive post-processing filter compared to the results in the slides

Filters used:
1. opening(5, 5)
2. closing(50, 20)
3. closing(20, 50)
4. opening(7, 7)

| Methods        | AP |
| ----------------  |-------------|
| FrameDifference    | 0.3759 |
| StaticFrameDifference    | 0.0956 |
| WeightedMovingMean    | 0.2940 |
| WeightedMovingVariance    | 0.3677 |
| AdaptiveBackgroundLearning    | 0.1101 |
| AdaptiveSelectiveBackgroundLearning    | 0.0976 |
| MixtureOfGaussianV2    | 0.1800 |
| DPAdaptiveMedian    | 0.2475 |
| DPGrimsonGMM    | 0.3083 |
| DPZivkovicAGMM    | 0.2643 |
| DPMean    | 0.2067 |
| DPWrenGA    | 0.2732 |
| DPPratiMediod    | 0.3508 |
| DPEigenbackground    | 0.1018 |
| T2FGMM_UM    | 0.0348 |

