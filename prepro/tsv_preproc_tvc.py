import os
import sys
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
import os.path as op
import json, yaml, code, io
import numpy as np
import pandas as pd
from src.utils.tsv_file_ops import tsv_writer
from src.utils.tsv_file_ops import generate_linelist_file
from collections import defaultdict

# data path to raw video files
data_vid_id = "datasets/TVC/raw_videos/{}/{}_clips/{}/{}.mp4"

# path to TVC repo and its annotations 
TVC_caption_anns = '/raid/keli/dataset/TVCaption/data/tvc_{}_release.jsonl'
TVC_subtitle_anns = '/raid/keli/dataset/TVCaption/data/tvqa_preprocessed_subtitles.jsonl'

# data_path: path to raw video files
data_path = "./datasets/TVC/{}/" 

dataset_path =  "./datasets/TVC/"
# To generate tsv files:
# {}.img.tsv: we use it to store video path info 
visual_file = "./datasets/TVC/{}.img.tsv" 
# {}.caption.tsv: we use it to store  captions
cap_file = "./datasets/TVC/{}.caption.tsv"
# {}.linelist.tsv: since each video may have multiple captions, we need to store the corresponance between vidoe id and caption id
linelist_file = "./datasets/TVC/{}.linelist.tsv"
# {}.label.tsv: we store any useful labels or metadara here, such as object tags. Now we only have captions. maybe can remove it in future. 
label_file = "./datasets/TVC/{}.label.tsv"

def write_to_yaml_file(context, file_name):
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, encoding='utf-8')

def tsv_reader(tsv_file, sep='\t'):
    with open(tsv_file, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]

def config_save_file(tsv_file, save_file=None, append_str='.new.tsv'):
    if save_file is not None:
        return save_file
    return op.splitext(tsv_file)[0] + append_str

def generate_caption_linelist_file(caption_tsv_file, save_file=None):
    num_captions = []
    for row in tsv_reader(caption_tsv_file):
        num_captions.append(len(json.loads(row[1])))

    cap_linelist = ['\t'.join([str(img_idx), str(cap_idx)]) 
            for img_idx in range(len(num_captions)) 
            for cap_idx in range(num_captions[img_idx])
    ]
    save_file = config_save_file(caption_tsv_file, save_file, '.linelist.tsv')
    with open(save_file, 'w') as f:
        f.write('\n'.join(cap_linelist))
    return save_file


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def load_json(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def dump_tsv_gt_to_coco_format(caption_tsv_file, outfile):
    annotations = []
    images = []
    cap_id = 0
    caption_tsv = tsv_reader(caption_tsv_file)

    for cap_row  in caption_tsv:
        image_id = cap_row[0]
        key = image_id
        caption_data = json.loads(cap_row[1])
        count = len(caption_data)
        for i in range(count):
            caption1 = caption_data[i]['caption']
            annotations.append(
                        {'image_id': image_id, 'caption': caption1,
                        'id': cap_id})
            cap_id += 1

        images.append({'id': image_id, 'file_name': key})

    with open(outfile, 'w') as fp:
        json.dump({'annotations': annotations, 'images': images,
                'type': 'captions', 'info': 'dummy', 'licenses': 'dummy'},
                fp)
 
def get_show_name(vid_name):
    """
    get tvshow name from vid_name
    :param vid_name: video clip name
    :return: tvshow name
    """
    show_list = ["friends", "met", "castle", "house", "grey"]
    vid_name_prefix = vid_name.split("_")[0]
    show_name = vid_name_prefix if vid_name_prefix in show_list else "bbt"
    return show_name

def get_show_season_name(show_name, vid_name):
    if show_name != 'bbt':
        season = vid_name.split('_')[1][0:3]
        return '{}_{}'.format(show_name, season)
    else:
        season = vid_name.split('_')[0][1:3]
        return '{}{}'.format(show_name, season)

def process_new(split):
    resolved_metadata_file = TVC_caption_anns.format(split)
    dataset = load_jsonl(resolved_metadata_file)

    asr_dict = {}
    subtitle = load_jsonl(TVC_subtitle_anns)
    for i in range(len(subtitle)):
        data = subtitle[i]
        key = data['vid_name']
        sub = data['sub']
        asr_dict[key] = sub

    img_label = []
    rows_label = []
    caption_label = []
    asr_no_match_counter = 0
    asr_match_counter = 0
    for i in range(len(dataset)):
        video_data = dataset[i]
        vid_name = video_data['vid_name']
        time_start = video_data['ts'][0]
        time_end = video_data['ts'][1]
        
        asr_data = asr_dict[vid_name]
        asr_text = ' '
        asr_available = False
        for j in range(len(asr_data)):
            asr_start = asr_data[j]['start']   
            asr_end = asr_data[j]['end']
            if int(asr_start)>=int(time_start) and int(asr_start)<=int(time_end):
                asr_text += asr_data[j]['text']
                asr_text += ' '
                asr_available = True
            asr_match_counter += 1
        if asr_available == False:
            asr_no_match_counter += 1

        tv_program = get_show_name(vid_name) 
        tv_prog_season = get_show_season_name(tv_program, vid_name)  
        if tv_program == 'bbt':
            resolved_data_vid_id = data_vid_id.format(tv_program+'_new', tv_program, tv_prog_season, vid_name)
        else:
            resolved_data_vid_id = data_vid_id.format(tv_program, tv_program, tv_prog_season, vid_name)
        resolved_data_vid_id = '{}_{}_{}'.format(resolved_data_vid_id, time_start, time_end)
        output_captions = []

        if 'descs' in video_data.keys():
            for desc in video_data['descs']:
                sentence = desc['desc']
                output_captions.append({"caption": sentence, "asr": asr_text, "start": time_start, "end": time_end})
        else:
            output_captions.append({"caption": ' ', "asr": asr_text, "start": time_start, "end": time_end})

        caption_label.append([str(resolved_data_vid_id),json.dumps(output_captions)]) 
        rows_label.append([str(resolved_data_vid_id),json.dumps(output_captions)]) 
        img_label.append([str(resolved_data_vid_id), str(resolved_data_vid_id)])            

    resolved_visual_file = visual_file.format(split)
    print("generating visual file for", resolved_visual_file)
    tsv_writer(img_label, resolved_visual_file)

    resolved_label_file = label_file.format(split)
    print("generating label file for", resolved_label_file)
    tsv_writer(rows_label, resolved_label_file)

    resolved_linelist_file = linelist_file.format(split)
    print("generating linelist file for", rows_label)
    generate_linelist_file(resolved_label_file, save_file=resolved_linelist_file)

    resolved_cap_file = cap_file.format(split)
    print("generating cap file for", resolved_cap_file)
    tsv_writer(caption_label, resolved_cap_file)
    print("generating cap linelist file for", resolved_cap_file)
    resolved_cap_linelist_file = generate_caption_linelist_file(resolved_cap_file)

    gt_file_coco = op.splitext(resolved_cap_file)[0] + '_coco_format.json'
    print("convert gt to", gt_file_coco)
    dump_tsv_gt_to_coco_format(resolved_cap_file, gt_file_coco)

    # print('asr_match_counter:', asr_match_counter)
    # print('asr_no_match_counter:', asr_no_match_counter)

    out_cfg = {}
    all_field = ['img', 'label', 'caption', 'caption_linelist', 'caption_coco_format']
    all_tsvfile = [resolved_visual_file, resolved_label_file, resolved_cap_file, resolved_cap_linelist_file, gt_file_coco]
    for field, tsvpath in zip(all_field, all_tsvfile):
        out_cfg[field] = tsvpath.split('/')[-1]
    out_yaml = '{}.yaml'.format(split)
    write_to_yaml_file(out_cfg, op.join(dataset_path, out_yaml))
    print('Create yaml file: {}'.format(op.join(dataset_path, out_yaml)))

def main():
    process_new('train')
    process_new('val')
    process_new('test_public')

if __name__ == '__main__':
    main()



