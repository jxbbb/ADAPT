# Assume in the docker container 
# EVAL_DIR='output_16frame_two/checkpoint-20-6600/'
EVAL_DIR='checkpoints/basemodel/checkpoints/'
CUDA_VISIBLE_DEVICES=0 python -m pdb src/tasks/run_caption_VidSwinBert.py \
       --val_yaml BDDX/testing_32frames.yaml  \
       --do_eval true \
       --do_train false \
       --eval_model_dir $EVAL_DIR