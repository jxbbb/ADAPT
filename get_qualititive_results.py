import json
import os

import cv2
import h5py
import numpy as np
from scipy import interpolate
from tqdm import tqdm


class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    HIGHL = '\x1b[6;30;42m'
    UNDERLINE = '\033[4m'


print(bcolors.WARNING + "demo for driving scene video caption" + bcolors.ENDC)

data_path = 'Videos/videos/053da4e3-48ec49ba.mov'
frame_per_caption = 20


def get_vid_info(file_obj):
    nFrames = int(file_obj.get(cv2.CAP_PROP_FRAME_COUNT))
    img_width = int(file_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(file_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = file_obj.get(cv2.CAP_PROP_FPS)

    return nFrames, img_width, img_height, fps


def get_frame(file_path=None):
    if file_path == None:
        str2read = data_path  # original image: 720x1280
    else:
        str2read = file_path
    frames = []
    cnt = 0
    scalefactor = 1
    if os.path.exists(str2read):

        video_capture = cv2.VideoCapture(str2read)
        nFrames, img_width, img_height, fps = get_vid_info(video_capture)

        print(bcolors.GREEN +
              'ID: {}, #Frames: {}, Image: {}x{}, FPS: {}'.format(
                  str2read, nFrames, img_width, img_height, fps) +
              bcolors.ENDC)

        for i in tqdm(range(nFrames)):
            gotImage, frame = video_capture.read()
            cnt += 1
            if gotImage:
                if cnt % 3 == 0:  # reduce to 10Hz
                    frame = frame.swapaxes(1, 0)
                    '''
                    if rotation > 0: 	frame = cv2.flip(frame,0)
                    elif rotation < 0: 	frame = cv2.flip(frame,1)
                    else: 				frame = frame.swapaxes(1,0)
                    print(frame.shape)
                    '''
                    frame = cv2.resize(frame,
                                       None,
                                       fx=0.125 * scalefactor,
                                       fy=0.125 * scalefactor)

                    if (frame.shape == (160 * scalefactor, 90 * scalefactor,
                                        3)):
                        frame = frame.swapaxes(1, 0)

                    try:
                        assert frame.shape == (90 * scalefactor,
                                               160 * scalefactor, 3)
                    except:
                        print(frame.shape)
                        exit()
                    # if cnt % 100 == 0:
                    #     cv2.imwrite('./demo/sample' + str(cnt) + '.png', frame)

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 640x360x3

                    frames.append(frame)

        video_capture.release()
    else:
        print(bcolors.FAIL +
              'ERROR: Unable to open video {}'.format(str2read) + bcolors.ENDC)
    frames = np.array(frames).astype(int)
    return frames

path = ""

all_cap = json.load(open("captions_BDDX.json", "r"))
all_cap = json.load(open("captions_BDDX.json", "r"))

caps = []
for anno in all_cap['annotations']:
    if anno['vidName'] == data_path.split(".")[0].split("/")[-1]:
        caps.append(anno)

log_path = "processed/log/" + data_path.split(".")[0].split("/")[-1] + '.h5'

if len(caps) != 0 and os.path.exists(log_path):
    create_video(caps, h5py.File(log_path))
else:
    print(log_path)
    print("do nothing")
