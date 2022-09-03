CUDA_VISIBLE_DEVICES=1 \
python -m pdb src/tasks/run_caption_VidSwinBert.py \
        --config src/configs/VidSwinBert/BDDX_multi_default.json \
        --train_yaml BDDX/training_32frames.yaml \
        --val_yaml BDDX/testing_32frames.yaml \
        --per_gpu_train_batch_size 4 \
        --per_gpu_eval_batch_size 16 \
        --num_train_epochs 40 \
        --learning_rate 0.0003 \
        --max_num_frames 32 \
        --pretrained_2d 0 \
        --backbone_coef_lr 0.05 \
        --mask_prob 0.5 \
        --max_masked_token 45 \
        --zero_opt_stage 1 \
        --mixed_precision_method deepspeed \
        --deepspeed_fp16 \
        --gradient_accumulation_steps 4 \
        --learn_mask_enabled \
        --loss_sparse_w 0.5 \
        --use_sep_cap \
        --multitask \
        --loss_sensor_w 0.1 \
        --output_dir ./expr/use_car_sensor