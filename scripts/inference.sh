# After launching the docker container 
EVAL_DIR='./results/inference/'
CHECKPOINT='expr/32frame/two/40_0.00025_0.04_0.4/checkpoint-14-4620/model.bin'
VIDEO='./docs/G0mjFqytJt4_000152_000162.mp4'
CUDA_VISIBLE_DEVICES=0 python src/tasks/run_caption_VidSwinBert_inference.py \
       --resume_checkpoint $CHECKPOINT  \
       --eval_model_dir $EVAL_DIR \
       --test_video_fname $VIDEO \
       --do_lower_case \
       --do_test 