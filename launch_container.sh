DATA_DIR=$1
MODEL_DIR=$2
OUTPUT=$3

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

if [ "$4" = "--prepro" ]; then
    RO=""
else
    RO=",readonly"
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/videocap,type=bind \
    --mount src=$DATA_DIR,dst=/videocap/datasets,type=bind$RO \
    --mount src=$MODEL_DIR,dst=/videocap/models,type=bind,readonly \
    --mount src=$OUTPUT,dst=/videocap/output,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /videocap linjieli222/videocap_torch1.7:fairscale \
    bash -c "source /videocap/setup.sh && bash" 
