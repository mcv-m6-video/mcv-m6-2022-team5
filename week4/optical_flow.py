import cv2
import math
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import inf

# load optical flow based on kitti dataset standard:
# Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
# contains the u-component, the second channel the v-component and the third
# channel denotes if a valid ground truth optical flow value exists for that
# pixel (1 if true, 0 otherwise). To convert the u-/v-flow into floating point
# values, convert the value to float, subtract 2^15 and divide the result by 64:
#
# flow_u(u,v) = ((float)I(u,v,1)-2^15)/64.0;
# flow_v(u,v) = ((float)I(u,v,2)-2^15)/64.0;
# valid(u,v)  = (bool)I(u,v,3);
def load_flow(path):
  flow = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.double)

  u_flow = (flow[:,:,2] - 2**15)/ 64
  v_flow = (flow[:,:,1] - 2**15)/ 64
  b_valid = flow[:,:,0]

  # # remove invalid points
  # u_flow[b_valid == 0] = 0
  # v_flow[b_valid == 0] = 0

  flow = [u_flow, v_flow, b_valid]
  return flow

def flow_error_distance(gt,kitti):
  return np.sqrt(np.square(gt[0] - kitti[0]) + np.square(gt[1] - kitti[1]))

# mean Square Error in Non-occluded areas
def flow_msen(gt, kitti):
    return np.mean(flow_error_distance(gt,kitti)[gt[2]==1])

# percentage of Erroneous Pixels in Non-occluded areas
def flow_pepn(gt, kitti, th=3):
    return 100 * (flow_error_distance(gt,kitti)[gt[2]==1] > th).sum() / (gt[2] != 0).sum()

def plot_optical_flow(flow):
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    images = [flow[0], flow[1], flow[2]]
    titles = ['u flow','v flow','b_valid']

    for ax,image,title in zip(axes.flatten(), images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.show()

# plot optical flow
def plot_optical_flow_field(img_path, flow, step=20, scale=0.1):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    plt.figure(figsize=(18,6))
    plt.imshow(img)

    h, w = flow[0].shape

    X = np.arange(0, w, step)
    Y = np.arange(0, h, step)
    U, V = np.meshgrid(X, Y)

    u_flow = flow[0][::step, ::step]
    v_flow = flow[1][::step, ::step]

    plt.quiver(U, V, u_flow, v_flow, np.hypot(u_flow, v_flow),scale_units='xy', angles='xy', scale=scale)
    plt.show()

def plot_error_distance(error_dis):
    plt.figure(figsize=(9, 3))
    plt.title('Square error visualised')
    plt.imshow(error_dis)
    plt.colorbar()

def plot_error_distribution_hist(error_dis, gt_flow):
    max_range = int(math.ceil(np.amax(error_dis)))

    plt.title('Mean square error distribution')
    plt.ylabel('Density')
    plt.xlabel('Mean square error')
    plt.hist(error_dis[gt_flow[2] == 1].ravel(), bins=30, range=(0.0, max_range))

def generate_motion_gif(filenames, save_path):
    with imageio.get_writer(save_path, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

def distance(patch1, patch2, method='ssd', weights=None):
    if weights is None:
        weights = np.ones(patch1.shape) / (patch1.shape[0]*patch1.shape[1])
    if method == 'ssd':
        diff = np.sum(weights * abs(patch1-patch2)**2)
    elif method == 'sad':
        diff = np.sum(weights * abs(patch1-patch2))
    elif method == 'ncc':
        product = (patch1 - patch1*weights) * (patch2 - patch2*weights)
        product = np.sum(weights * product)
        stds1 = np.sqrt(np.sum(weights*np.square(patch1 - patch1*weights)))
        stds2 = np.sqrt(np.sum(weights*np.square(patch2 - patch2*weights)))
        stds = stds1 * stds2
        if stds == 0:
            return 0
        else:
            product /= stds
        diff = -product #making negative the value is equivalint to look for the maximun
    return diff


def block_matching(current_img, past_img, proccess='backward', metric='ssd',N=16, P=16, stride=1):
    if proccess == 'backward':
        target_img = current_img
        ref_img = past_img
    else:
        ref_img = current_img
        target_img = past_img

    h, w, _ = ref_img.shape
    optical_flow = np.zeros((h, w, 2))
    for rows in tqdm(range(0, h, N)):
        for cols in range(0, w, N):
            block = ref_img[rows:rows+N,cols:cols+N]
            area_minx = max(0, cols - P)
            area_miny = max(0, rows - P)
            area_maxX = min(w, cols + N + P)
            area_maxy = min(h, rows + N + P)
            # area_target = target_img[area_miny:area_maxy, area_minx:area_maxX]
            blocH_size = block.shape[0]
            blocW_size = block.shape[1]                
            minDist = inf
            for y in range(area_miny, area_maxy - N, stride):
                for x in range(area_minx, area_maxX - N, stride):
                    dist = distance(block, target_img[y:y+blocH_size, x:x+blocW_size], metric)
                    if dist < minDist:
                        minDist = dist
                        optical_flow[rows:rows+N,cols:cols+N] = [cols - x, rows - y]
    
    optical_flow = np.moveaxis(optical_flow, -1, 0)
    if proccess == 'backward':
        return (-optical_flow)
    return optical_flow


def compute_block_of(block, box, im_target, mode, P, stride=1):
    blocH_size = block.shape[0]
    blocW_size = block.shape[1]                
    area_minx = max(0, int(box[0]) - P)
    area_miny = max(0, int(box[1]) - P)
    area_maxX = min(im_target.shape[1] - blocW_size, int(box[2]) + P)
    area_maxy = min(im_target.shape[0] - blocH_size, int(box[3]) + P)
    # area_target = target_img[area_miny:area_maxy, area_minx:area_maxX]
    minDist = inf
    of = []
    for y in (range(area_miny, area_maxy-blocH_size, stride)):
        for x in range(area_minx, area_maxX-blocW_size, stride):
            dist = distance(block, im_target[y:y+blocH_size, x:x+blocW_size], 'ssd')
            if dist < minDist:
                minDist = dist
                of = np.array([x - box[0], y - box[1]])
    if mode == 'forward':
        return -of
    return of
    
