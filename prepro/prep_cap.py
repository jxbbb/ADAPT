#|**********************************************************************;
# Project           : Explainable Deep Driving
#
# File name         : Step1_csv2json.py
#
# Author            : Jinkyu Kim
#
# Date created      : 20181201
#
# Purpose           : Download .mov files of BDD-V dataset
#
# Revision History  :
#
# Date        Author      Ref    Revision
# 20181201    jinkyu      1      initiated
#
# Remark
#|**********************************************************************;

from logging import raiseExceptions
from    scipy           import interpolate
import  json
import  csv
from    sys             import platform
from    tqdm            import tqdm
import  numpy           as np
import  os
import  time
import  glob
import  cv2
import  scipy.misc
import  h5py
from    random          import shuffle
import  skvideo.io
import  skvideo.datasets
from    scipy.ndimage   import rotate
from    src.utils       import *
import time

# Main function
#----------------------------------------------------------------------
if __name__ == "__main__":
    if platform == 'linux':
        config = dict2(**{
            "annotations":  '/data/hdd01/jinbu/BDDX/BDD-V/BDD-X-Annotations_v1.csv', # (video url, start, end, action, justification)
            "save_path":    './data/processed/',
            "data_path":    '../../BDDX/BDD-V/Videos/',
            "chunksize":    10 })
    else:
        raise NotImplementedError

    # JSON format
    data = {}
    data['annotations'] = []
    data['info']        = []
    data['videos']      = []

    # Parameters
    maxItems = 15 # Each csv file contains action / justification pairs of 15 at maximum

    # output path
    if not os.path.exists(config.save_path+"log/"): os.makedirs(config.save_path+"log/")
    if not os.path.exists(config.save_path+"cam_new/"): os.makedirs(config.save_path+"cam_new/")

    # Read information about video clips
    with open(config.annotations) as f_obj:
        examples = csv_dict_reader(f_obj)

        '''
        Keys:
            1. Input.Video, 2. Answer.1start, 3. Answer.1end, 4. Answer.1action, 5. Answer.1justification
        '''
        captionid   = -1
        videoid     = -1
        vidNames    = []
        vidNames_notUSED = []
        for item in examples:
            # time.sleep(2)
            vidName  = item['Input.Video'].split("/")[-1][:-4]

            if len(vidName)==0: 
                vidNames_notUSED.append(str(videoid) + "_" + str(vidName))
                continue   
            if len(item["Answer.1start"])==0: 
                vidNames_notUSED.append(str(videoid) + "_" + str(vidName))
                continue
            if len(item["Answer.1justification"])==0: 
                vidNames_notUSED.append(str(videoid) + "_" + str(vidName))
                continue       
            videoid += 1  

            #print(bcolors.HEADER + "Video: {}".format(vidName) + bcolors.ENDC)
            

            # #--------------------------------------------------
            # # 1. Control signals
            # #--------------------------------------------------
            # str2find  = '%sinfo/100k/train/%s.json'%(config.data_path, vidName)
            # json2read = glob.glob(str2find)

            # if json2read: 
            #     json2read = json2read[0]
            # else: 
            #     print( bcolors.FAIL + "Unable to read json file: {}".format(str2find) + bcolors.ENDC )
            #     vidNames_notUSED.append(str(videoid) + "_" + str(vidName))
            #     continue

            #     # MuteVideo data set seems to have different label format.
            #     # Will use first eight codes
            #     #json2read = glob.glob('%sinfo/*/*%s*.json'%(config.data_path, vidName[:8]))
            #     #if json2read:
            #     #    json2read = json2read[0]
            #     #else:
            #     #    print( bcolors.FAIL + "Unable to read json file: {}".format(str2find) + bcolors.ENDC )
            #     #    vidNames_notUSED.append(str(videoid) + "_" + str(vidName))
            #     #    continue

            # # keys: timestamp, longitude, course, latitude, speed, accuracy
            # timestamp, longitude, course, latitude, speed, accuracy, gps_x, gps_y = [], [], [], [], [], [], [], []
            # with open(json2read) as json_data:
            #     trajectories = json.load(json_data)['locations']
            #     for trajectory in trajectories:
            #         timestamp.append(trajectory['timestamp'])
            #         longitude.append(trajectory['longitude'])
            #         course.append(trajectory['course'])
            #         latitude.append(trajectory['latitude'])
            #         speed.append(trajectory['speed'])
            #         accuracy.append(trajectory['accuracy'])

            #         # gps to flatten earth coordinates (meters)
            #         _x, _y, _ = lla2flat( (trajectory['latitude'], trajectory['longitude'], 1000.0),
            #         							 (latitude[0], longitude[0]), 0.0, -100.0)
            #         gps_x.append(_x)
            #         gps_y.append(_y)


            # # Use interpolation to prevent variable periods
            # if np.array(timestamp).shape[0] < 2:
            #     print(bcolors.FAIL + "Timestamp is not VALID: {}".format(str2find) + bcolors.ENDC)
            #     continue

            # # extract equally-spaced points
            # points, dist_steps, cumulative_dist_along_path = get_equally_spaced_points( gps_x, gps_y )

            # # Generate target direction
            # goalDirection_equal  = get_goalDirection( dist_steps, points )
            # goalDirection_interp = interpolate.interp1d(dist_steps, goalDirection_equal)
            # goalDirection        = goalDirection_interp(cumulative_dist_along_path)

            # # Generate curvatures / accelerator
            # curvature_raw    = compute_curvature(points[0], points[1])
            # curvature_interp = interpolate.interp1d(dist_steps, curvature_raw)
            # curvature        = curvature_interp(cumulative_dist_along_path)
            # accelerator 	 = np.gradient(speed)

            # #print(bcolors.GREEN + "Processed >> Ego-motions: {} sequences".format(len(timestamp)) + bcolors.ENDC)





            #--------------------------------------------------
            # 2. Captions
            #--------------------------------------------------
            nEx = 0
            video_annotations = []
            for segment in range(maxItems-1):
                sTime           = item["Answer.{}start".format(segment+1)]
                eTime           = item["Answer.{}end".format(segment+1)]
                action          = item["Answer.{}action".format(segment+1)]
                justification   = item["Answer.{}justification".format(segment+1)]

                if not sTime or not eTime or not action or not justification: continue

                nEx         += 1
                captionid   += 1

                # Info 
                feed_dict = { 'contributor':    'Berkeley DeepDrive',
                              'date_created':   time.strftime("%d/%m/%Y"),
                              'description':    'This is 0.1 version of the BDD-X dataset',
                              'url':            'https://deepdrive.berkeley.edu',
                              'year':           2017}
                data['info'].append(feed_dict)

                # annotations
                feed_dict = { 'action':         action,
                              'justification':  justification,
                              'sTime':          sTime,
                              'eTime':          eTime,
                              'id':             captionid,
                              'vidName':        vidName,
                              'video_id':       videoid,
                              }
                data['annotations'].append(feed_dict)
                video_annotations.append(feed_dict)

                # Video
                feed_dict = { 'url':            item['Input.Video'],
                              'video_name':     vidName,
                              'height':         720,
                              'width':          1280,
                              'video_id':       videoid,
                               }

                data['videos'].append(feed_dict)

            print(bcolors.GREEN + "Processed >> Annotations: {} sub-examples".format(nEx) + bcolors.ENDC)








            # #--------------------------------------------------
            # # 3. Read video clips
            # #--------------------------------------------------
            # str2read = '%svideos/%s.mov'%(config.data_path, vidName) # original image: 720x1280
            # frames   = []
            # cnt      = 0
            # scalefactor = 0.5

            # if (os.path.isfile(config.save_path+"cam_new/"+ str(videoid).zfill(5) + "_" + str(vidName) + ".h5")) == True: 
            #     print(bcolors.GREEN + 
            #         'File already generated (decoding): {}'.format(str(videoid) + "_" + str(vidName)) 
            #         + bcolors.ENDC)
            #     continue
            
            # elif os.path.exists(str2read):
            #     metadata = skvideo.io.ffprobe(str2read)

            #     if ("side_data_list" in metadata["video"].keys()) == False:
            #         rotation = 0
            #     else:
            #         rotation = float(metadata["video"]["side_data_list"]["side_data"]["@rotation"])

            #     cap = cv2.VideoCapture(str2read)
            #     nFrames, img_width, img_height, fps = get_vid_info(cap)

            #     print(bcolors.GREEN + 
            #         'ID: {}, #Frames: {}, Image: {}x{}, FPS: {}'.format(vidName, nFrames, img_width, img_height, fps) 
            #         + bcolors.ENDC)

            #     for i in tqdm(range(nFrames)):
            #         gotImage, frame = cap.read()
            #         cnt += 1
            #         if gotImage:
            #             if cnt%1==0:
            #                 # frame = frame.swapaxes(1,0)
            #                 # '''
            #                 # if rotation > 0: 	frame = cv2.flip(frame,0)
            #                 # elif rotation < 0: 	frame = cv2.flip(frame,1)
            #                 # else: 				frame = frame.swapaxes(1,0)
            #                 # print(frame.shape)
            #                 # '''
            #                 # frame = cv2.resize(frame, None, fx=scalefactor, fy=scalefactor)

            #                 # if(frame.shape == (720*scalefactor, 1280*scalefactor, 3)):
            #                 #     frame = frame.swapaxes(1,0)

            #                 # try:
            #                 #     assert frame.shape == (1280*scalefactor, 720*scalefactor, 3)
            #                 # except:
            #                 #     print(frame.shape)
            #                 #     exit()
            #                 # # if cnt%100==0:
            #                 # #     #cv2.imshow('image', frame)
            #                 # #     #cv2.waitKey(10)
            #                 # #     if videoid>2000:
            #                 # #         cv2.imwrite('../exam_png/sample'+str(cnt)+'_'+str(videoid)+'.png',frame)

            #                 # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 640x360x3

            #                 frames.append(frame)

            #     cap.release()
            # else:
            #     print(bcolors.FAIL + 
            #         'ERROR: Unable to open video {}'.format(str2read) 
            #         + bcolors.ENDC)
            #     break

            # # frames = np.array(frames).astype(int)

            # s_frame = -1
            # e_frame = -1
            # las_s_frame = -2
            # las_e_frame = -2
            # for anno in video_annotations:
            #     video_id_save = anno['video_id']
            #     vidName_old = anno['vidName']
            #     caption_id_save = anno['id']
            #     s_frame = int(int(anno['sTime'])*fps)
            #     e_frame = int(int(anno['eTime'])*fps+fps)
            #     if s_frame > e_frame:
            #         print(f"s_frame{s_frame} > e_frame{e_frame}", anno['vidName'])
            #         exit()
            #     if e_frame-s_frame <= 1:
            #         print(f"e_frame{e_frame}-s_frame{s_frame} <= 1", anno['vidName'])
            #         exit()
            #     if nFrames - int(s_frame) < 65 :
            #         print(f"{s_frame} > nFrames{nFrames}", anno['vidName'])
            #         exit()
            #     frame_choose = np.linspace(s_frame, min(e_frame, nFrames), num=64, endpoint=False, retstep=False, dtype=np.int32)
            #     print("frame_choose", frame_choose)
            #     for save_order, chose_num in enumerate(frame_choose):
            #         frame_to_save = frames[chose_num]
            #         save_root_path = "data/processed/64frame/"
            #         if not os.path.exists(save_root_path):
            #             os.mkdir(save_root_path)
            #         vidName_save = str(video_id_save).zfill(5) + '_' + str(vidName_old) + '_' + str(caption_id_save).zfill(2)
            #         frame64_save_path = os.path.join(save_root_path, vidName_save)
            #         if not os.path.exists(frame64_save_path):
            #             os.mkdir(frame64_save_path)
            #         cv2.imwrite(frame64_save_path+f"/{str(save_order).zfill(3)}.jpg", frame_to_save)
            # las_s_frame = s_frame
            # las_e_frame = e_frame



            # #--------------------------------------------------
            # # 4. Saving
            # #--------------------------------------------------
            # vidNames.append(str(videoid) + "_" + str(vidName))
            
            # if (os.path.isfile(config.save_path+"cam_new/"+ str(videoid).zfill(5) + "_" + str(vidName) + ".h5")) == False: 
            #     cam_new = h5py.File(config.save_path+ "cam_new/" + str(videoid).zfill(5)  + "_" + str(vidName) + ".h5", "w")
            #     # dset = cam_new.create_dataset("/X",         data=frames,   chunks=(config.chunksize,1280*scalefactor,720*scalefactor,3), dtype='uint8')
            # else:
            #     print(bcolors.GREEN + 
            #         'File already generated (cam_new): {}'.format(str(videoid) + "_" + str(vidName)) 
            #         + bcolors.ENDC)
            
            # # if (os.path.isfile(config.save_path+"log/"+ str(videoid) + "_" + str(vidName) + ".h5")) == False: 
            # #     log = h5py.File(config.save_path+ "log/" + str(videoid) + "_" + str(vidName) + ".h5", "w")
            
            # #     dset = log.create_dataset("/timestamp", data=timestamp)
            # #     dset = log.create_dataset("/longitude", data=longitude)
            # #     dset = log.create_dataset("/course",    data=course)
            # #     dset = log.create_dataset("/latitude",  data=latitude)
            # #     dset = log.create_dataset("/speed",     data=speed)
            # #     dset = log.create_dataset("/accuracy",  data=accuracy)
            # #     #dset = log.create_dataset("/fps",       data=fps)
            # #     dset = log.create_dataset("/curvature",  data=curvature, 	 dtype='float')
            # #     dset = log.create_dataset("/accelerator",data=accelerator, 	 dtype='float')
            # #     dset = log.create_dataset("/goaldir",    data=goalDirection, dtype='float')

            # # else:
            # #     print(bcolors.GREEN + 
            # #         'File already generated (log): {}'.format(str(videoid) + "_" + str(vidName)) 
            # #         + bcolors.ENDC)
  
    with open(config.save_path+'captions_BDDX.json', 'w') as outfile:  
        json.dump(data, outfile)

    np.save(config.save_path + 'vidNames_notUSED.txt', vidNames_notUSED)