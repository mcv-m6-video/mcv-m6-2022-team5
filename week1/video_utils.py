import cv2
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

def generate_videoBB(videoPath, detections, videoName='videoBoundingBox'):
    fig, ax = plt.subplots()

    vidcap = cv2.VideoCapture(videoPath)
    _, image = vidcap.read()
    ims = []
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame in tqdm(range(1,num_frames)):
        _, image = vidcap.read()
        if str(frame) in detections:
            for b in detections[str(frame)]:
                tl = (int(b.xtl), int(b.ytl))
                br = (int(b.xbr), int(b.ybr))
                color = (0,255,0)
                image = cv2.rectangle(image, tl, br, color, 2)
            im = ax.imshow(image, animated=True)
            ims.append([im])
    # break

    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=10000)
    ani.save(videoName + ".gif")