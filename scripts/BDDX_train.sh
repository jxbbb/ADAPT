CUDA_VISIBLE_DEVICES=0 \
python -m pdb src/tasks/run_caption_VidSwinBert.py \
        --config src/configs/VidSwinBert/BDDX_8frm_default.json \
        --train_yaml BDDX/training_32frames.yaml \
        --val_yaml BDDX/validation_32frames.yaml \
        --per_gpu_train_batch_size 1 \
        --per_gpu_eval_batch_size 1 \
        --num_train_epochs 40 \
        --learning_rate 0.0003 \
        --max_num_frames 16 \
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
        # --use_sep_cap \
        --output_dir ./output_16frame_one