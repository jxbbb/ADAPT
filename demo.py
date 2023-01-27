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

path = "expr/multitask/sensor_course/checkpoint-36-9216/"
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



def visulize(des, exp):

    assert des['image_id'] == exp['image_id']

    vid_id = des['image_id'].split('_')[-1]

    str2read = data_path  # original image: 720x1280
    frames = []
    cnt = 0
    scalefactor = 1
    if os.path.exists(str2read):

        video_capture = cv2.VideoCapture(str2read)
        nFrames, img_width, img_height, fps = get_vid_info(video_capture)

        print(bcolors.GREEN + "create_video" + bcolors.ENDC)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # avi格式
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')#MP4格式
        out = cv2.VideoWriter('./demo/demo.mp4', fourcc, 30,
                              (img_width, img_height))

        for _ in tqdm(range(nFrames)):
            gotImage, frame = video_capture.read()
            cnt += 1
            if gotImage:
                cap = ""
                for anno in caps:
                    if cnt >= int(anno['sTime'])*30 and cnt < int(anno['eTime'])*30:
                        cap = anno['action'] + ' ' + anno['justification']
                        break

                accelerator, accuracy, course, curvature, goaldir, latitude, longitude, speed = infos['accelerator'], infos['accuracy'], infos['course'], infos['curvature'], infos['goaldir'], infos['latitude'], infos['longitude'], infos['speed']

                new_fps = nFrames / accelerator.shape[0]

                dist_steps = [int(new_fps*i) for i in range(accelerator.shape[0])]

                accelerator_interp = interpolate.interp1d(dist_steps, accelerator)
                course_interp = interpolate.interp1d(dist_steps, course)
                curvature_interp = interpolate.interp1d(dist_steps, curvature)
                goaldir_interp = interpolate.interp1d(dist_steps, goaldir)
                latitude_interp = interpolate.interp1d(dist_steps, latitude)
                longitude_interp = interpolate.interp1d(dist_steps, longitude)
                speed_interp = interpolate.interp1d(dist_steps, speed)
                # print(dist_steps[-1])
                sec_order = min(cnt, dist_steps[-1])

                accelerator = round(float(accelerator_interp(sec_order)), 2)
                course = round(float(course_interp(sec_order)), 2)
                curvature = round(float(curvature_interp(sec_order)), 2)
                goaldir = round(float(goaldir_interp(sec_order)), 2)
                latitude = round(float(latitude_interp(sec_order)), 2)
                longitude = round(float(longitude_interp(sec_order)), 2)
                speed = round(float(speed_interp(sec_order)), 2)


                # sec_order = min(cnt//30, accelerator.shape[0]-1)

                # accelerator = round(accelerator[sec_order], 2)
                # accuracy = round(accuracy[sec_order], 2)
                # course = round(course[sec_order], 2)
                # curvature = round(curvature[sec_order], 2)
                # goaldir = round(goaldir[sec_order], 2)
                # latitude = round(latitude[sec_order], 2)
                # longitude = round(longitude[sec_order], 2)
                # speed = round(speed[sec_order], 2)
                info_txt_1 = 'v:'+str(speed).zfill(5) + ' a:'+ str(accelerator).zfill(5)
                info_txt_2 = 'c:'+str(course).zfill(5) + ' cu:'+ str(curvature).zfill(5) + ' gd:'+str(goaldir).zfill(5)
                info_txt_3 = 'gps:'+str(latitude).zfill(5)+','+str(longitude).zfill(5)

                cv2.putText(frame, info_txt_1, (50, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (238, 238, 66), 2)

                cv2.putText(frame, info_txt_2, (50, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (238, 238, 66), 2)

                cv2.putText(frame, info_txt_3, (50, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (238, 238, 66), 2)

                cv2.putText(frame, cap, (100, 600), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (238, 238, 66), 2)

                out.write(frame)
                # if cnt % 100 == 0:
                #     cv2.imwrite('./demo/sample' + str(cnt) + '.png', frame)

        video_capture.release()
        out.release()
    else:
        print(bcolors.FAIL +
              'ERROR: Unable to open video {}'.format(str2read) + bcolors.ENDC)
    frames = np.array(frames).astype(int)
    return frames


all_des = json.load(open(path+"pred.BDDX_des.testing_32frames.beam1.max15_coco_format", "r"))
all_exp = json.load(open(path+"pred.BDDX_exp.testing_32frames.beam1.max15_coco_format", "r"))

for i in range(len(all_des)):
    visulize(all_des[i], all_exp[i])

