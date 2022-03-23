import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from PIL import Image
from eval_utils import *
from video_utils import *
from load_utils import *
from background_remover import *
import pybgs as bgs

data_path = '../datasets/AICity_data/train/S03/c010/'

def generate_gif(videoPath, fgbg, videoName='video'):
    fig, ax = plt.subplots()
    plt.axis('off')
    
    vidcap = cv2.VideoCapture(videoPath)
    ims = []
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame in tqdm(range(1,num_frames//4)):
        for i in range(4):
            _, image = vidcap.read()
        fgmask = fgbg.apply(image)
        
        im = ax.imshow(fgmask, animated=True)
        ims.append([im])
    # break

    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=10000)
    ani.save(videoName + ".gif", writer=animation.PillowWriter(fps=24))

def getBoxesFromMask2(mask):
    # output = cv2.connectedComponentsWithStats(np.uint8(mask), 8, cv2.CV_32S)
    # (numLabels, labels, boxes, centroids) = output
    counts, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    detectedElems = []
    for cont in counts: #First box is always the background
        x,y,w,h = cv2.boundingRect(cont)
        if w*h > 700: #Try to do a better estimation of the minimunm size
            b = VehicleDetection(0, -1, float(x), float(y), float(w), float(h), float(-1))
            detectedElems.append(b)

    return detectedElems

def remove_background3(videoPath, ROIpath, fgbg):
    roi = cv2.imread(ROIpath, cv2.IMREAD_GRAYSCALE)
    
    vidcap = cv2.VideoCapture(videoPath)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = {}
    for frame in tqdm(range(num_frames)):
        _, image = vidcap.read()
        if frame >= num_frames // 4:

            # image = cv2.medianBlur(image, 7)
            fgmask = fgbg.apply(image)
            fgmask[fgmask==127]=0

            roi_applied = cv2.bitwise_and(fgmask, roi)
            cleaned = opening(roi_applied, 5, 5)  # initial removal of small noise
            cleaned = closing(cleaned, 50, 20)  # vertical filling of areas [SWITCH TO HORIZONTAL?]
            cleaned = closing(cleaned, 20, 50)  # vertical filling of areas [SWITCH TO HORIZONTAL?]
            cleaned = opening(cleaned, 7, 7)

#             cv2.imwrite(f'./masks/mask_{frame}.png', roi_applied)

            detections[str(frame)] = getBoxesFromMask2(cleaned)
            
    return detections

### GT without parked cars
gt_detect = readDetectionsXML('ai_challenge_s03_c010-full_annotation.xml')
gt_notParked = {}
for frame, objs in gt_detect.items():
    obj_notParked = []
    for ob in objs:
        if not ob.parked:
            obj_notParked.append(ob)
    if len(obj_notParked) > 0:
        gt_notParked[frame] = obj_notParked

## bgslibrary algorithms
algorithms=[]
algorithms.append(bgs.FrameDifference())
algorithms.append(bgs.StaticFrameDifference())
algorithms.append(bgs.WeightedMovingMean())
algorithms.append(bgs.WeightedMovingVariance())
algorithms.append(bgs.AdaptiveBackgroundLearning())
algorithms.append(bgs.AdaptiveSelectiveBackgroundLearning())
algorithms.append(bgs.MixtureOfGaussianV2())
algorithms.append(bgs.DPAdaptiveMedian())
algorithms.append(bgs.DPGrimsonGMM())
algorithms.append(bgs.DPZivkovicAGMM())
algorithms.append(bgs.DPMean())
algorithms.append(bgs.DPWrenGA())
algorithms.append(bgs.DPPratiMediod())
algorithms.append(bgs.DPEigenbackground())
algorithms.append(bgs.DPTexture())
algorithms.append(bgs.T2FGMM_UM())
algorithms.append(bgs.T2FGMM_UV())
algorithms.append(bgs.T2FMRF_UM())
algorithms.append(bgs.T2FMRF_UV())
algorithms.append(bgs.FuzzySugenoIntegral())
algorithms.append(bgs.FuzzyChoquetIntegral())
algorithms.append(bgs.LBSimpleGaussian())
algorithms.append(bgs.LBFuzzyGaussian())
algorithms.append(bgs.LBMixtureOfGaussians())
algorithms.append(bgs.LBAdaptiveSOM())
algorithms.append(bgs.LBFuzzyAdaptiveSOM())
algorithms.append(bgs.LBP_MRF())
algorithms.append(bgs.MultiLayer())
algorithms.append(bgs.PixelBasedAdaptiveSegmenter())
algorithms.append(bgs.VuMeter())
algorithms.append(bgs.KDE())
algorithms.append(bgs.IndependentMultimodal())
algorithms.append(bgs.MultiCue())
algorithms.append(bgs.SigmaDelta())
algorithms.append(bgs.SuBSENSE())
algorithms.append(bgs.LOBSTER())
algorithms.append(bgs.PAWCS())
algorithms.append(bgs.TwoPoints())
algorithms.append(bgs.ViBe())
algorithms.append(bgs.CodeBook())

# for method in []

for algorithm in algorithms:
    algorithm_name = str(algorithm.__class__)[13:-2]
    print("Running ", algorithm_name)


    detections = remove_background3(data_path + 'vdo.avi', data_path + 'roi.jpg', algorithm)
    rec, prec, ap, tp_gauss, IoU_tp, IoU = voc_eval(gt_notParked, detections, 0.5, False)
    plot_prec_recall_curve(prec, rec, f'Precision-Recall curve for {algorithm_name} - AP {ap:.2f}', 'algorithms/'+algorithm_name+'.png')
    np.save('algorithms/'+algorithm_name+'.npy',[prec,rec])
    print(algorithm_name, '_AP: ',ap)