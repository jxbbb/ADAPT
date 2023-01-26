# After launching the docker container 
EVAL_DIR='/videocap/expr/16frame/two/40_0.0003_0.05_0.5/checkpoint-23-5060/'
CHECKPOINT='/videocap/expr/16frame/two/40_0.0003_0.05_0.5/checkpoint-23-5060/model.bin'
VIDEO='/videocap/241e8319-4ca76d61.mov'
CUDA_VISIBLE_DEVICES=0 python  /videocap/src/tasks/run_caption_VidSwinBert_inference.py \
       --resume_checkpoint $CHECKPOINT  \
       --eval_model_dir $EVAL_DIR \
       --test_video_fname $VIDEO \
       --do_lower_case \
       --do_test 
