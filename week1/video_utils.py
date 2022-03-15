import cv2
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from PIL import Image

def drawBoxes(image, predictions, color):
    for b in predictions:
        tl = (int(b.xtl), int(b.ytl))
        br = (int(b.xbr), int(b.ybr))
        image = cv2.rectangle(image, tl, br, color, 2)

    return image

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



def generate_videoBB_comparison(videoPath, gt, predicted, videoName='videoBoundingBox', initialFrame=1, lastFrame=100):
    vidcap = cv2.VideoCapture(videoPath)
    ims = []
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame in tqdm(range(num_frames)):
        sucess, image = vidcap.read()
        if not sucess:
            break
        elif frame >= initialFrame and frame <= lastFrame:
            if str(frame) in gt:
                color = (0,255,0)
                image = drawBoxes(image, gt[str(frame)], color)

                if str(frame) in predicted:
                    color = (255,0,0)
                    image = drawBoxes(image, predicted[str(frame)], color)
                print('frame number:', frame)
                ims.append(Image.fromarray(image))
        elif frame > lastFrame:
            break
    print('frames',len(ims))
    frame_one = ims[0]
    frame_one.save(videoName + ".gif", format="GIF", append_images=ims,
               save_all=True, duration=30, loop=0)