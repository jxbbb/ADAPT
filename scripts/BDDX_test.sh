# Assume in the docker container 
EVAL_DIR='output_8frame/checkpoint-9-4000/'
CUDA_VISIBLE_DEVICES=6 python src/tasks/run_caption_VidSwinBert.py \
       --val_yaml BDDX/testing_32frames.yaml  \
       --do_eval true \
       --do_train false \
       --eval_model_dir $EVAL_DIR