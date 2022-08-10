# Assume in the docker container 
EVAL_DIR='output_32frame_des/checkpoint-13-5720'
CUDA_VISIBLE_DEVICES=0 python src/tasks/run_caption_VidSwinBert.py \
       --val_yaml BDDX_des/testing_32frames.yaml  \
       --do_eval true \
       --do_train false \
       --eval_model_dir $EVAL_DIR