ln -s /evalcap/coco_caption /videocap/src/evalcap/coco_caption
ln -s /evalcap/cider /videocap/src/evalcap/cider
pip install fvcore ete3 transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade azureml-core -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install visualize -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ffmpeg-python -i https://pypi.tuna.tsinghua.edu.cn/simple
df -h
ls -al
export TORCH_HOME=/models
