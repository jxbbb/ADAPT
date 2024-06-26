export REPO_DIR=$PWD
DATA_DIR=$REPO_DIR'/datasets/'
DATA_PART_DIR=$REPO_DIR'/datasets_part/'
MODEL_DIR=$REPO_DIR'/models/'
OUTPUT=$REPO_DIR'/output/'


if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

if [ "$1" = "--prepro" ]; then
    RO=""
else
    RO=",readonly"
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/videocap,type=bind \
    --mount src=$DATA_DIR,dst=/videocap/datasets,type=bind$RO \
    --mount src=$DATA_PART_DIR,dst=/videocap/datasets_part,type=bind$RO \
    --mount src=$MODEL_DIR,dst=/videocap/models,type=bind,readonly \
    --mount src=$OUTPUT,dst=/videocap/output,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /videocap jxbbb/adapt:latest \
    bash -c "source /videocap/setup.sh && bash" 
