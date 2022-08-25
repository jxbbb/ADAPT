import os
from os.path import join
from tqdm import tqdm
import multiprocessing as mp
import subprocess
import datetime
import  json

def get_video_duration(video_file):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", video_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)


def extract_frame_from_video(video_path, save_frame_path, fps=1, num_frames=-1,
                             start_ts=-1, end_ts=-1,
                             suppress_msg=False, other_args="", overwrite=True):
    """Uniformly split a video into clips of length {clip_len}.
    i.e., in the case of clip_len=60, the clips will be 00:00:00-00:01:00, 00:01:00-00:02:00, etc, ...

    Note that we drop the first (usually opening remark, etc) and last (ask for subscription, etc) clip.

    Args:
        video_path:
        save_frame_path:
        fps: frame_per_second, default 1
        suppress_msg:
        other_args: str, other ffmpeg args, such as re-scale to 720p with '-vf scale=-1:720'

    Returns:

    """
    extra_args = " -hide_banner -loglevel panic " if suppress_msg else ""
    extra_args += " -y " if overwrite else ""
    if start_ts != -1 and end_ts != -1:
        
        dur_to_use = get_video_duration(video_path)
        if end_ts > dur_to_use:
            if start_ts > dur_to_use:
                print("start_ts > dur_to_use")
                exit()
            else:
                end_ts = int(dur_to_use)
        if int(end_ts - start_ts) == 0:
            if start_ts < 2:
                end_ts += 2
            elif dur_to_use-end_ts < 2:
                start_ts -= 2
            else:
                start_ts -= 1
                end_ts   += 1
        elif int(end_ts - start_ts) == 1:
            if start_ts < 2:
                end_ts   += 1
            else:
                start_ts -= 1
        else:
            pass

        start_ts_str = str(datetime.timedelta(seconds=start_ts))
        end_ts_str = str(datetime.timedelta(seconds=end_ts))
        duration = str(datetime.timedelta(seconds=(end_ts - start_ts)))
        

            
        
        # print(start_ts, end_ts, duration)
        extra_args += f"-ss {start_ts_str} -t {duration} "
    # extra_args2 = " -vf scale=720:-2 "
    # -preset veryfast:  (upgrade to latest ffmpeg if error)
    # https://superuser.com/questions/490683/cheat-sheets-and-presets-settings-that-actually-work-with-ffmpeg-1-0
    if num_frames <= 0 :
        split_cmd_template = "ffmpeg {extra} -i {video_path} -vf fps={fps} {output_frame_path}%06d.jpg"
        
        cur_split_cmd = split_cmd_template.format(
            extra=extra_args, video_path=video_path, fps=fps, output_frame_path=save_frame_path)
    else:
        # get duration of the video
        if start_ts != -1 and end_ts != -1:
            duration = end_ts - start_ts
        else:
            duration = get_video_duration(video_path)
        if duration <= 0:
            duration = 10
            print(video_path)
        frame_rate = num_frames/duration

        # if not suppress_msg:
        #     print(duration, frame_rate, num_frames)
        output_exists = True
        for frame_idx in range(num_frames):
            if not os.path.exists(f"{save_frame_path}{(frame_idx+1):04d}.jpg"):
                print(f"{save_frame_path}{(frame_idx+1):04d}.jpg does not exist")
                output_exists = False
                # save_frame_path = save_frame_path.replace(f"{num_frames}frames", f"{num_frames}frames_debug")

                break
        if output_exists:
            return
        split_cmd_template = "ffmpeg {extra} -i {video_path} -vf fps={frame_rate} {output_frame_path}%04d.jpg"

        cur_split_cmd = split_cmd_template.format(
            extra=extra_args, video_path=video_path, frame_rate=frame_rate, output_frame_path=save_frame_path)
        if not suppress_msg:
            print(cur_split_cmd)
    try:
        _ = subprocess.run(cur_split_cmd.split(), stdout=subprocess.PIPE)
    except Exception as e:
        print(f"Error returned by ffmpeg cmd {e}")


COMMON_VIDEO_ETX = set([
    ".webm", ".mpg", ".mpeg", ".mpv", ".ogg",
    ".mp4", ".m4p", ".mpv", ".avi", ".wmv", ".qt",
    ".mov", ".flv", ".swf"])


def extract_frame(video_info, save_dir, fps, num_frames, debug=False, corrupt_files=[]):
    (video_file_path, sTime, eTime, caption_id) = video_info
    filename = os.path.basename(video_file_path)
    vid = os.path.splitext(filename)[0]
    frame_name = f"{vid}_frame"
    frame_save_dir = join(save_dir, caption_id)
    frame_save_path = join(frame_save_dir, frame_name)

    launch_extract = True
    if launch_extract:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(frame_save_dir, exist_ok=True)
        # scale=width:height
        extract_frame_from_video(video_file_path, frame_save_path, fps=fps, num_frames=num_frames,
                                 start_ts=sTime, end_ts=eTime,
                                 suppress_msg=not debug, other_args="")


def load_tsv_to_mem(tsv_file, sep='\t'):
    data = []
    with open(tsv_file, 'r') as fp:
        for _, line in enumerate(fp):
            data.append([x.strip() for x in line.split(sep)])
    return data


def extract_all_frames(video_root_dir, save_dir, fps, num_frames,
                       video_info_tsv, corrupt_files, num_workers, debug=False):

    with open(video_info_tsv) as f_obj:
        raw_video_info = json.load(f_obj)


        videoFiles = []
        for _, line_item in enumerate(raw_video_info['annotations']):
            vidName = line_item['vidName']
            sTime = int(line_item['sTime'])
            eTime = int(line_item['eTime'])
            caption_id = str(line_item['id']).zfill(5)
            
            input_file = video_root_dir+vidName+'.mov'
            # input_file = input_file.replace('datasets','_datasets')
            if os.path.isfile(input_file):
                videoFiles.append((input_file, sTime, eTime, caption_id))
        if debug:
            videoFiles = videoFiles[:1]

        if num_workers > 0:
            from functools import partial
            extract_frame_partial = partial(
                extract_frame, fps=fps,
                save_dir=save_dir, debug=debug, corrupt_files=corrupt_files,
                num_frames=num_frames)

            with mp.Pool(num_workers) as pool, tqdm(total=len(videoFiles)) as pbar:
                for idx, _ in enumerate(
                        pool.imap_unordered(
                            extract_frame_partial, videoFiles, chunksize=8)):
                    pbar.update(1)
        else:
            for idx, d in tqdm(enumerate(videoFiles),
                            total=len(videoFiles), desc="extracting frames from video"):
                extract_frame(d, save_dir, fps=fps, debug=debug,corrupt_files=corrupt_files, num_frames=num_frames)


def main():

    video_root_dir = "/data/hdd01/jinbu/BDDX/BDD-V/Videos/videos/"
    save_dir = 'data/32frames'
    video_info_tsv = '/data/hdd01/jinbu/video_preprocess/code/data/processed/captions_BDDX.json'
    extract_all_frames(video_root_dir, save_dir, 1,
                       32, video_info_tsv, corrupt_files=None,
                       num_workers=16, debug=False)


if __name__ == '__main__':
    main()