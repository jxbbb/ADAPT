import os
import sys
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
import os.path as op
import json, yaml, code, io
import numpy as np
from src.utils.tsv_file_ops import tsv_reader, tsv_writer
from src.utils.tsv_file_ops import generate_linelist_file
from tqdm import tqdm

# data path to raw video files
data_vid_id = "datasets/MSVD/raw_videos/{}"
dataset_path = './datasets/MSVD/'
# annotations downloaded from official downstream dataset
MSVD_anns = './datasets/MSVD/msvd-retrieval_train-val-test.json'

# To generate tsv files:
# {}.img.tsv: we use it to store video path info 
visual_file = "./datasets/MSVD/{}.img.tsv"
# {}.caption.tsv: we use it to store  captions
cap_file = "./datasets/MSVD/{}.caption.tsv"
# {}.linelist.tsv: since each video may have multiple captions, we need to store the corresponance between vidoe id and caption id
linelist_file = "./datasets/MSVD/{}.linelist.tsv"
# {}.label.tsv: we store any useful labels or metadara here, such as object tags. Now we only have captions. maybe can remove it in future.
label_file = "./datasets/MSVD/{}.label.tsv"

def write_to_yaml_file(context, file_name):
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, encoding='utf-8')

def tsv_reader(tsv_file, sep='\t'):
    with open(tsv_file, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]

def csv_reader(csv_file, sep=','):
    with open(csv_file, 'r') as fp:
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
 
def process(split):

    import json
    data = json.load(open(MSVD_anns, 'r'))[split]

    img_label = []
    rows_label = []
    caption_label = []
    from collections import defaultdict
    vid2captions = defaultdict(list)
    for d_ in tqdm(data):
        videoid, caption = d_["video"], d_["caption"]
        resolved_data_vid_id = data_vid_id.format(f"{videoid}.avi")
        if not op.exists(resolved_data_vid_id):
            continue
        
        vid2captions[resolved_data_vid_id].append({"caption": caption})
    
    for resolved_data_vid_id, output_captions in tqdm(vid2captions.items()):
        caption_label.append([str(resolved_data_vid_id),json.dumps(output_captions)]) 
        rows_label.append([str(resolved_data_vid_id),json.dumps(output_captions)])
        img_label_item = [str(resolved_data_vid_id), str(resolved_data_vid_id)]
        if img_label_item not in img_label_item:
            img_label.append(img_label_item)

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
    out_yaml = '{}.yaml'.format(split)
    write_to_yaml_file(out_cfg, op.join(dataset_path, out_yaml))
    print('Create yaml file: {}'.format(op.join(dataset_path, out_yaml)))


def main():
    process('train')
    process('val')
    process('test')

if __name__ == '__main__':
    main()

