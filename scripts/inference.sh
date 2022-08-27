# After launching the docker container 
EVAL_DIR='expr/32frame/two/40_0.0003_0.05_0.4/checkpoint-40-13210/'
CHECKPOINT='expr/32frame/two/40_0.0003_0.05_0.4/checkpoint-40-13210/model.bin'
VIDEO='1f693cf9-ef7a83ca.mov'
CUDA_VISIBLE_DEVICES=0 python -m pdb src/tasks/run_caption_VidSwinBert_inference.py \
       --resume_checkpoint $CHECKPOINT  \
       --eval_model_dir $EVAL_DIR \
       --test_video_fname $VIDEO \
       --do_lower_case \
       --do_test 