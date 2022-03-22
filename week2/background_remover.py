import cv2
from VehicleDetection import VehicleDetection
import numpy as np
from tqdm import tqdm

def closing(mask, kernel_w=3, kernel_h=3):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)

def opening(mask, kernel_w=3, kernel_h=3):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)

def getBoxesFromMask(mask):
    counts, hier = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    detectedElems = []
    for cont in counts: #First box is always the background
        x,y,w,h = cv2.boundingRect(cont)
        if w*h > 700: #Try to do a better estimation of the minimunm size
            b = VehicleDetection(0, -1, float(x), float(y), float(w), float(h), float(-1))
            detectedElems.append(b)

    return detectedElems

def cleanMask(mask, roi):
    roi_applied = cv2.bitwise_and(mask, roi)
    cleaned = opening(roi_applied, 5, 5) #initial removal of small noise
    cleaned = closing(cleaned, 50, 20) #horizontal filling of areas [SWITCH TO HORIZONTAL?]
    cleaned = closing(cleaned, 20, 50) #vertical filling of areas [SWITCH TO HORIZONTAL?]
    cleaned = opening(cleaned, 7, 7)

    return cleaned


def remove_background(means, stds, videoPath, ROIpath, alpha=4, sigma=2):
    roi = cv2.imread(ROIpath, cv2.IMREAD_GRAYSCALE)
    
    vidcap = cv2.VideoCapture(videoPath)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = {}
    for frame in tqdm(range(num_frames)):
        _, image = vidcap.read()
        if frame >= num_frames // 4:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            img_mask = np.zeros(img_gray.shape, dtype = np.uint8)
            img_mask[abs(img_gray - means) >= alpha * (stds + sigma)] = 255

            cleaned = cleanMask(img_mask, roi)

            cv2.imwrite(f'./masks/mask_{frame}.png', cleaned)

            detections[str(frame)] = getBoxesFromMask(cleaned)

    return detections

def remove_background_adaptative(means, stds, videoPath, ROIpath, alpha=4, sigma=2, p=0.1):
    roi = cv2.imread(ROIpath, cv2.IMREAD_GRAYSCALE)
    vidcap = cv2.VideoCapture(videoPath)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    detections = {}
    for frame in tqdm(range(num_frames)):
        _, image = vidcap.read()
        if frame >= num_frames // 4:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_mask = np.zeros(img_gray.shape, dtype=np.uint8)
            img_mask[abs(img_gray - means) >= alpha * (stds + sigma)] = 255

            cleaned = cleanMask(img_mask, roi)
            cv2.imwrite(f'./masks_adaptative/mask_{frame}.png', cleaned)

            #update mean and std
            idxs = cleaned == 0
            means[idxs] = p * img_gray[idxs] + (1 - p) * means[idxs]
            stds[idxs] = np.sqrt(p * (img_gray[idxs] - means[idxs])**2 + (1 - p) * stds[idxs]**2)

            detections[str(frame)] = getBoxesFromMask(cleaned)

    return detections

def get_background_stats(videoPath, initFrame=1, lastFrame=514):
    vidcap = cv2.VideoCapture(videoPath)
    _, image = vidcap.read()

    ims_for_stats = lastFrame - initFrame + 1
    ims = np.zeros((ims_for_stats, image.shape[0], image.shape[1]))

    ims[0,:,:] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for frame in tqdm(range(initFrame, lastFrame)):
        _, image = vidcap.read()
        ims[frame,:,:] = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    means = np.mean(ims, axis=0)
    # means = np.median(ims, axis=0)
    stds = np.std(ims, axis=0)
    return means, stds