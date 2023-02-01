# Assume in the docker container 
EVAL_DIR='expr/multitask/sensor_course/checkpoint-8-2048/'
CUDA_VISIBLE_DEVICES=0 python src/tasks/run_caption_VidSwinBert.py \
       --val_yaml BDDX/testing_32frames.yaml  \
       --do_eval true \
       --do_train false \
       --do_signal_eval \
       --eval_model_dir $EVAL_DIR