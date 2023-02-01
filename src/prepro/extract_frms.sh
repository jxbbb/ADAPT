# python -m pdb ./prepro/extract_BDDV_frames.py \
# --video_root_dir ./datasets/YouCook2/raw_videos/training/ \
# --save_dir ./datasets/YouCook2/ \
# --video_info_tsv ./datasets/YouCook2/training.img.tsv \
# --num_frames 32 \
# --debug

python ./prepro/extract_frames.py

python ./prepro/create_image_frame_tsv.py \
    --dataset YouCook2 \
    --split training \
    --image_size 256 \
    --num_frames 32

python ./prepro/tsv_preproc_BDDX.py