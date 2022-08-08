import csv
import json
all_info = json.load(open("datasets/BDDX/captions_BDDX.json"))
annos = all_info['annotations']
with open(r"./validation.caption.linelist.tsv", 'w', newline='') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    i = 0
    for anno in annos:
        # if anno['id'] > 21142 and anno['id']<= 23661:
        if anno['id'] > 21142 and anno['id']<= 23661:
            # tsv_w.writerow([anno['vidName'], [anno]])
            tsv_w.writerow([i])
            i+=1
