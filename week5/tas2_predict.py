import argparse
def parse_args():
    parser = argparse.ArgumentParser(description= 'Arguments to run the inference script')
    parser.add_argument('-v', '--video', default=None, type=str, help='Absolute path to video to extract images')
    parser.add_argument('-o', '--out_path', default='./results', type=str, help='Relative path to output folder')
    parser.add_argument('-m', '--model', default='COCO-Detection/retinanet_R_101_FPN_3x.yaml', type=str, help='Detectron2 Model')
    parser.add_argument('-d', '--detections', default='model_detections', type=str, help='Name of the file to save the detections')
    parser.add_argument('-l', '--load_w', default=None, type=str, help='Path to trained weights')
    parser.add_argument('-c', '--pred_class', default=[0], type=list, help='Class to predict')

    return parser.parse_args()

args = parse_args()
from glob import glob
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, cv2, pickle

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from PIL import Image
import argparse
from VehicleDetection import VehicleDetection
import time



video_path = args.video

config_file = args.model

os.makedirs(args.out_path, exist_ok=True)

############################
#   INFERENCE
############################

#Then, we create a detectron2 config and a detectron2 DefaultPredictor to run inference on this image.
cfg = get_cfg()

## MASK RCNN
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
if args.load_w is not None:
    cfg.MODEL.WEIGHTS = args.load_w
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
else:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
predictor = DefaultPredictor(cfg)

vidcap = cv2.VideoCapture(video_path)
num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

model_detections = {}

start = time.time()
for frame in range(num_frames):
    _, im = vidcap.read()
    outputs = predictor(im)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    # print(outputs["instances"].pred_boxes)
    # car_instances = outputs["instances"][outputs["instances"].pred_classes == args.pred_class]

    # FIX HARDCODING
    car_instances = outputs["instances"][outputs["instances"].pred_classes == 2]
    truck_instances = outputs["instances"][outputs["instances"].pred_classes == 7]

    # car_instances = outputs["instances"][outputs["instances"].pred_classes in args.pred_class] #REVISE!

    # We can use `Visualizer` to draw the predictions on the image.
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
    # out = v.draw_instance_predictions(car_instances.pred_classes == 2].to("cpu"))
    # image = Image.fromarray(out.get_image()[:, :, ::-1])
    # image.save(f'{args.out_path}/predicted_{frame}.png')
    model_detections[str(frame)] = []
    # break
    for id in range(len(car_instances)):
        box = car_instances.pred_boxes[id].tensor.to('cpu').detach().numpy()[0]
        # print(box)
        vh = VehicleDetection(frame, -1, 
                            float(box[0]), float(box[1]), 0, 0, 
                            car_instances.scores[id].to('cpu').numpy(), float(box[2]), float(box[3]))
        model_detections[str(frame)].append(vh)

    for id in range(len(truck_instances)):
        box = truck_instances.pred_boxes[id].tensor.to('cpu').detach().numpy()[0]
        # print(box)
        vh = VehicleDetection(frame, -1, 
                            float(box[0]), float(box[1]), 0, 0, 
                            car_instances.scores[id].to('cpu').numpy(), float(box[2]), float(box[3]))
        model_detections[str(frame)].append(vh)
    

end = time.time()
print(f'Elapsed time to infer all video frames for model {args.detections}: {end - start}')
with open(args.detections + '.pkl', "wb") as output_file:
    pickle.dump(model_detections, output_file)

####################################
