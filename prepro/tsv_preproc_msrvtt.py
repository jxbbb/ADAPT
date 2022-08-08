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
data_vid_id = "datasets/MSRVTT-v2/raw_videos/{}.mp4"

# dataset_path: path to dataset
dataset_path = "./datasets/MSRVTT-v2/"
# annotations downloaded from official downstream dataset
json_path = "./datasets/MSRVTT-v2/MSRVTT_data.json"
train_csv = "./datasets/MSRVTT-v2/MSRVTT_train.9k.csv"
test_csv = "./datasets/MSRVTT-v2/MSRVTT_JSFUSION_test.csv"

# To generate tsv files:
# {}.img.tsv: we use it to store video path info 
visual_file = "./datasets/MSRVTT-v2/{}.img.tsv"
# {}.caption.tsv: we use it to store  captions
cap_file = "./datasets/MSRVTT-v2/{}.caption.tsv"
# {}.linelist.tsv: since each video may have multiple captions, we need to store the corresponance between vidoe id and caption id
linelist_file = "./datasets/MSRVTT-v2/{}.linelist.tsv"
# {}.label.tsv: we store any useful labels or metadara here, such as object tags. Now we only have captions. maybe can remove it in future.
label_file = "./datasets/MSRVTT-v2/{}.label.tsv"



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

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def load_json(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def mk_video_ret_datalist(raw_datalist):
    """
    Args:
        raw_datalist: list(dict)
        cfg:
    Returns:
    """

    datalist = []
    qid = 0
    for raw_d in raw_datalist:
        d = dict(
            id=qid,
            txt=raw_d["caption"],
            vid_id=raw_d["clip_name"]
        )
        qid += 1
        datalist.append(d)

    return datalist

def process(split_type, csv_path):
    # structs we need
    img_label = []
    rows_label = []
    caption_label = []

    gt_csv = pd.read_csv(csv_path)
    gt_data = json.load(open(json_path, 'r'))

    assert split_type in ["train", "val", "test"]
    # Train: video0 : video6512 (6513)
    # Val: video6513 : video7009 (497)
    # Test: video7010 : video9999 (2990)
    video_ids = [gt_data['videos'][idx]['video_id'] for idx in range(len(gt_data['videos']))]
    split_dict = {"train": video_ids[:6513], "val": video_ids[6513:6513 + 497], "test": video_ids[6513 + 497:]}
    choiced_video_ids = split_dict[split_type]

    sample_len = 0
    sentences_dict = {}
    video_sentences_dict = defaultdict(list)
    if split_type == "train":  # expand all sentence to train
        for itm in gt_data['sentences']:
            if itm['video_id'] in choiced_video_ids:
                sentences_dict[len(sentences_dict)] = (itm['video_id'], itm['caption'])
                video_sentences_dict[itm['video_id']].append(itm['caption'])
    elif split_type == "val" or split_type == "test":
        for itm in gt_data['sentences']:
            if itm['video_id'] in choiced_video_ids:
                video_sentences_dict[itm['video_id']].append(itm['caption'])
        for vid in choiced_video_ids:
            sentences_dict[len(sentences_dict)] = (vid, video_sentences_dict[vid][0])
    else:
        raise NotImplementedError

    sample_len = len(sentences_dict)

    for vid in choiced_video_ids: 
        captions = video_sentences_dict[vid]
        num_sample = len(captions)
        if num_sample > 0:
            output_captions = []
            labels = []
            for i in range(num_sample):
                resolved_data_vid_id = data_vid_id.format(vid)
                data_txt = captions[i]
                output_captions.append({"caption": data_txt})

            caption_label.append([str(resolved_data_vid_id),json.dumps(output_captions)]) 
            rows_label.append([str(resolved_data_vid_id),json.dumps(output_captions)]) 
            img_label.append([str(resolved_data_vid_id), str(resolved_data_vid_id)])

    split = split_type
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

    out_cfg = {}
    all_field = ['img', 'label', 'caption', 'caption_linelist', 'caption_coco_format']
    all_tsvfile = [resolved_visual_file, resolved_label_file, resolved_cap_file, resolved_cap_linelist_file, gt_file_coco]
    for field, tsvpath in zip(all_field, all_tsvfile):
        out_cfg[field] = tsvpath.split('/')[-1]
    out_yaml = '{}.yaml'.format(split_type)
    write_to_yaml_file(out_cfg, op.join(dataset_path, out_yaml))
    print('Create yaml file: {}'.format(op.join(dataset_path, out_yaml)))

def main():
    process('train', train_csv)
    process('val', train_csv)
    process('test', test_csv)

if __name__ == '__main__':
    main()


