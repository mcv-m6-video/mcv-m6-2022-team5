import motmetrics as mm
import numpy as np

def create_accumulator():
    # Creates accumulator

    return mm.MOTAccumulator(auto_id=True)

def update_accumulator(acc, gt_object_ids, det_object_ids, IoUs):
    # Updates accumulator with info from current frame
    #   gt_object_ids: List with ground truth object ids in the frame
    #   det_object_ids: List with detection object ids in the frame
    #   IoUs: For every ground truth object id, IoU scores relative to every detection (list of lists)

    acc.update(
        gt_object_ids,
        det_object_ids,
        IoUs
    )

    return acc

def display_metrics(acc, selected_metrics = ['num_frames', 'mota', 'motp', 'idf1']):
    # Computes and displays multiple object tracking metrics 

    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=selected_metrics, name='acc')
    print(summary)

    

# # Example below [IGNORE]

# # Create an accumulator that will be updated during each frame
# acc = mm.MOTAccumulator(auto_id=True)

# acc.update(
#     [1, 2],                     # Ground truth objects in this frame
#     [1, 2, 3],                  # Detector hypotheses in this frame
#     [
#         [0.1, np.nan, 0.3],     # Distances from object 1 to hypotheses 1, 2, 3
#         [0.5,  0.2,   0.3]      # Distances from object 2 to hypotheses 1, 2, 3
#     ]
# )

# print(acc.mot_events)

# frameid = acc.update(
#     [1, 2],
#     [1],
#     [
#         [0.2],
#         [0.4]
#     ]
# )

# print(acc.mot_events)

# frameid = acc.update(
#     [1, 2],
#     [1, 3],
#     [
#         [0.6, 0.2],
#         [0.1, 0.6]
#     ]
# )

# print(acc.mot_events)

# mh = mm.metrics.create()

# summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1'], name='acc')
# print(summary)