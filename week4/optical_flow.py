import cv2
import math
import imageio
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

