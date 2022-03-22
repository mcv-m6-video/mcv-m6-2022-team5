import cv2
from VehicleDetection import VehicleDetection
import numpy as np
from tqdm import tqdm

def cleanMask(mask, kernel_size=3):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
    # erosion_dst = cv2.erode(mask, element)
    # dilation = cv2.dilate(erosion_dst, element)
    # mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)
    # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, element)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)

def getBoxesFromMask(mask):
    output = cv2.connectedComponentsWithStats(np.uint8(mask), 8, cv2.CV_32S)
    (numLabels, labels, boxes, centroids) = output
    detectedElems = []
    for box in boxes[1:]: #First box is always the background
        if box[4] > 500: #Try to do a better estimation of the minimunm size
            b = VehicleDetection(0, -1, float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(-1))
            detectedElems.append(b)

    return detectedElems

def remove_background(means, stds, videoPath, alpha=4, sigma=2, kernelMorph=5):
    vidcap = cv2.VideoCapture(videoPath)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = {}
    for frame in tqdm(range(num_frames)):
        _, image = vidcap.read()
        if frame >= num_frames // 4:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_mask = np.zeros(img_gray.shape)
            img_mask[abs(img_gray - means) >= alpha * (stds + sigma)] = 255

            cleaned = cleanMask(img_mask, 7)
            cv2.imwrite(f'./masks/mask_{frame}.png', cleaned)

            detections[str(frame)] = getBoxesFromMask(cleaned)

    return detections

def remove_background_adaptative(means, stds, videoPath, alpha=4, sigma=2, p=0.1):
    vidcap = cv2.VideoCapture(videoPath)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = {}
    for frame in tqdm(range(num_frames)):
        _, image = vidcap.read()
        if frame >= num_frames // 4:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_mask = np.zeros(img_gray.shape)
            img_mask[abs(img_gray - means) >= alpha * (stds + sigma)] = 255

            cleaned = cleanMask(img_mask, 7)
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