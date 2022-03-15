import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

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
  v_flow = -(flow[:,:,1] - 2**15)/ 64
  b_valid = flow[:,:,0]

  # remove invalid points
  u_flow[b_valid == 0] = 0
  v_flow[b_valid == 0] = 0

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

def plot_error_distance(error_dis):
    plt.figure(figsize=(9, 3))
    plt.title('Error distance')
    plt.colorbar()
    plt.imshow(error_dis)

def plot_error_distribution_hist(error_dis, gt_flow):
    max_range = int(math.ceil(np.amax(error_dis)))

    plt.title('Error distribution')
    plt.ylabel('Density')
    plt.xlabel('Error distance')
    plt.hist(error_dis[gt_flow[2] == 1].ravel(), bins=30, range=(0.0, max_range))

