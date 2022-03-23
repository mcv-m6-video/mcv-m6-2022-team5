import cv2
from VehicleDetection import VehicleDetection
import numpy as np
from tqdm import tqdm
import copy

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
        
        if 29 < w < 593 and 12 < h < 442: #Condition based on GT minimums* and maximums
            if 0.4 < w/h < 2.5: #Condition to avoid too elongated boxes
                b = VehicleDetection(0, -1, float(x), float(y), float(w), float(h), float(-1))
                detectedElems.append(b)

    return detectedElems

def cleanMask(mask, roi):
    roi_applied = cv2.bitwise_and(mask, roi)
    cleaned = opening(roi_applied, 5, 5) #initial removal of small noise
    cleaned = closing(cleaned, 2, 80) #vertical filling of areas
    cleaned = closing(cleaned, 80, 2) #horizontal filling of areas
    # cleaned = closing(cleaned, 40, 1) #2nd horizontal filling of areas
    cleaned = closing(cleaned, 2, 80) #general filling
    cleaned = opening(cleaned, 10, 10) #final cleaning

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
            
            img_gray = cv2.medianBlur(img_gray, 7)
            
            img_mask = np.zeros(img_gray.shape, dtype = np.uint8)
            img_mask[abs(img_gray - means) >= alpha * (stds + sigma)] = 255

            cleaned = cleanMask(img_mask, roi)
            
            cv2.imwrite(f'./masks/mask_{frame}.png', cleaned)

            detections[str(frame)] = getBoxesFromMask(cleaned)
            
            cleaned = cv2.cvtColor(cleaned,cv2.COLOR_GRAY2RGB)
            for b in detections[str(frame)]:
                tl = (int(b.xtl), int(b.ytl))
                br = (int(b.xbr), int(b.ybr))
                color = (255,0,0)
                cleaned = cv2.rectangle(cleaned, tl, br, color, 2)
            cv2.imwrite(f'./masks_bb/mask_{frame}.png', cleaned)

    return detections


def remove_background_adaptative(means, stds, videoPath, ROIpath, alpha=4, sigma=2, p=0.04):
    roi = cv2.imread(ROIpath, cv2.IMREAD_GRAYSCALE)
    vidcap = cv2.VideoCapture(videoPath)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    meansV = copy.deepcopy(means)
    stdsV = copy.deepcopy(stds)


    detections = {}
    for frame in tqdm(range(num_frames)):
        _, image = vidcap.read()
        if frame >= num_frames // 4:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
#             img_gray = cv2.medianBlur(img_gray, 7)
            
            img_mask = np.zeros(img_gray.shape, dtype=np.uint8)
            img_mask[abs(img_gray - meansV) >= alpha * (stdsV + sigma)] = 255
            
            cleaned = cleanMask(img_mask, roi)
            cv2.imwrite(f'./masks_adaptative/mask_{frame}.png', cleaned)

            detections[str(frame)] = getBoxesFromMask(cleaned)

            #update mean and std
            idxs = (img_mask == 0)
            meansV[idxs] = p * img_gray[idxs] + (1 - p) * meansV[idxs]
            stdsV[idxs] = np.sqrt(p * (img_gray[idxs] - meansV[idxs])**2 + (1 - p) * stdsV[idxs]**2)


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
    stds = np.std(ims, axis=0)
    return means, stds


def get_background_stats_color(videoPath, initFrame=1, lastFrame=514, color=cv2.COLOR_BGR2RGB):
    vidcap = cv2.VideoCapture(videoPath)
    _, image = vidcap.read()

    ims_for_stats = lastFrame - initFrame + 1
    ims = np.zeros((ims_for_stats, image.shape[0], image.shape[1], 3), dtype = "float32")

    ims[0,:,:,:] = cv2.cvtColor(image, color)
    for frame in tqdm(range(initFrame, lastFrame)):
        _, image = vidcap.read()
        ims[frame,:,:,:] = (cv2.cvtColor(image, color))

    means_C1 = np.mean(ims[:,:,:,0], axis=0)
    means_C2 = np.mean(ims[:,:,:,1], axis=0)
    means_C3 = np.mean(ims[:,:,:,2], axis=0)

    stds_C1 = np.std(ims[:,:,:,0], axis=0)
    stds_C2 = np.std(ims[:,:,:,1], axis=0)
    stds_C3 = np.std(ims[:,:,:,2], axis=0)

    return [means_C1,means_C2,means_C3], [stds_C1,stds_C2,stds_C3]