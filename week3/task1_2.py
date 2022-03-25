from asyncio import selector_events
from importlib.resources import path
from load_utils  import *
from os import  listdir

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import some common libraries
import numpy as np
import os, argparse, cv2
import pickle

# Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from PIL import Image

from customization_support import get_KITTIMOTS_dicts
# ------------------------------------------------------------

# Preparing the custom dataset
def parse_args():
    parser = argparse.ArgumentParser(description= 'Arguments to run the inference script')
    parser.add_argument('-g', '--ground', default='../../m6_dataset/ai_challenge_s03_c010-full_annotation.xml', type=str)
    parser.add_argument('-p', '--img_path', default='../../m6_dataset/dataset/', type=str)
    parser.add_argument('-o', '--out_path', default='./results', type=str, help='Relative path to output folder')
    parser.add_argument('-m', '--model', default='COCO-Detection/retinanet_R_101_FPN_3x.yaml', type=str, help='Detectron2 Model')
    parser.add_argument('-d', '--detections', default='model_detections', type=str, help='Name of the file to save the detections')

    return parser.parse_args()

args = parse_args()
selected_model = args.model
path_train_imgs = args.img_path
format_path = path_train_imgs + 'vdo_{}.png'

gt_detected = readDetectionsXML(args.ground)
frames = listdir(path_train_imgs)

SAVEPATH = './KITTIMOTS_m6_dict'

for d, data in [('train', frames[0:len(frames)//4]), ('valid', frames[len(frames)//4:])]:
    DatasetCatalog.register("KITTIMOTS_" + d, lambda d=data: get_KITTIMOTS_dicts(gt_detected,d,format_path))
    MetadataCatalog.get("KITTIMOTS_" + d).set(thing_classes=["car"])
KITTIMOTS_metadata = MetadataCatalog.get("KITTIMOTS_train")


# Loading or saving KITTIMOTS dicts
saving_enabled = True
saved_KITTIMOTS_dicts = SAVEPATH + '.pkl'

if os.path.exists(saved_KITTIMOTS_dicts):
    with open(saved_KITTIMOTS_dicts, 'rb') as reader:
        print('Loading dataset dicts...')
        dataset_dicts = pickle.load(reader)
else:
    print('Generating dict for training...')
    dataset_dicts = get_KITTIMOTS_dicts('train')
    if saving_enabled == True:
        with open(saved_KITTIMOTS_dicts, 'wb') as handle:
            print('Saving dataset dicts...')
            pickle.dump(dataset_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Example visualization (representative case)

img_filename = path_train_imgs + 'vdo_1232.png'
for element in dataset_dicts:
    if element['file_name'] == img_filename:
        d = element
img = cv2.imread(img_filename)
im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
visualizer = Visualizer(im_rgb[:, :, ::-1], metadata=KITTIMOTS_metadata, scale=1.2)
out = visualizer.draw_dataset_dict(d)
image = Image.fromarray(out.get_image()[:, :, ::-1])
image.save('detectron2_GT.png',)

# ------------------------------------------------------------

# Training

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(selected_model))
cfg.DATASETS.TRAIN = ("KITTIMOTS_train",)
cfg.DATASETS.VAL = ('KITTIMOTS_valid',)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(selected_model)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = 6000
cfg.SOLVER.STEPS = [] # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

cfg.OUTPUT_DIR = args.out_path

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# ------------------------------------------------------------

# Inference and evaluation

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# Example of inference on significative image sample
img = cv2.imread(img_filename)
im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

outputs = predictor(im_rgb)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
v = Visualizer(im_rgb[:, :, ::-1], metadata=KITTIMOTS_metadata, scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
image = Image.fromarray(out.get_image()[:, :, ::-1])
image.save('detectron2_trained.png',)

# Evaluation based on COCO metrics

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("KITTIMOTS_valid", output_dir=args.out_path)
val_loader = build_detection_test_loader(cfg, "KITTIMOTS_valid")
print(inference_on_dataset(predictor.model, val_loader, evaluator))