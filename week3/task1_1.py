from glob import glob
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from PIL import Image
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description= 'Arguments to run the inference script')
    parser.add_argument('-v', '--video', default=None, type=str, help='Absolute path to video to extract images')
    parser.add_argument('-e', '--extension', default='.jpg', type=str, help='Absolute path to image folder')
    parser.add_argument('-o', '--out_path', default='./results_out_of_context', type=str, help='Relative path to output folder')
    parser.add_argument('-m', '--model', default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', type=str, help='Detectron2 Model')

    return parser.parse_args()

args = parse_args()

dataset_path = args.path

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
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
predictor = DefaultPredictor(cfg)

vidcap = cv2.VideoCapture(args.video)
num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

for frame in range(num_frames):
    if frame == 200:
        break
    _, im = vidcap.read()
    # im = np.array(Image.open(filename))
    outputs = predictor(im)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)
    cars = outputs["instances"].pred_classes == 2

    dict_to_print = {'pred_classes': outputs["instances"].pred_classes[cars], 'pred_boxes': outputs["instances"].pred_boxes[cars],
                    'scores': outputs["instances"].scores[cars]}


    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
    out = v.draw_instance_predictions(dict_to_print.to("cpu"))
    image = Image.fromarray(out.get_image()[:, :, ::-1])
    image.save(f'{args.out_path}/predicted_{frame}.png')

####################################
