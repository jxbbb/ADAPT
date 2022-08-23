import json

with open("/data/hdd01/jinbu/video_preprocess/code/data/processed/captions_BDDX.json") as f:
    data = json.load(f)
    for anno in data['annotations']:
        if int(anno['sTime']) > int(anno['eTime']):
            print(anno['vidName'], anno['sTime'] , anno['eTime'])
            exit()
     