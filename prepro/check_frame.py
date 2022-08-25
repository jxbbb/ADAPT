import os


root_path = "/data/hdd01/jinbu/video_preprocess/code/data/32frames"
all_num = len(os.listdir(root_path))

for i in range(all_num):
    frame_path = os.path.join(root_path, str(i).zfill(5))
    frame_num = len(os.listdir(frame_path))
    if frame_num <  32:
        print("frame_num < 31", i)
        break