
python ./prepro/extract_frames.py

python ./prepro/create_image_frame_tsv.py \
    --dataset YouCook2 \
    --split training \
    --image_size 256 \
    --num_frames 32

python ./prepro/tsv_preproc_BDDX.py