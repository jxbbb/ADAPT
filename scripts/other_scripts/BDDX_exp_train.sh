CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMPI_COMM_WORLD_SIZE="4" \
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=25678 src/tasks/run_adapt.py \
        --config src/configs/VidSwinBert/BDDX_8frm_default.json \
        --train_yaml BDDX_exp/training_32frames.yaml \
        --val_yaml BDDX_exp/testing_32frames.yaml \
        --per_gpu_train_batch_size 6 \
        --per_gpu_eval_batch_size 16 \
        --num_train_epochs 10 \
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
        --output_dir ./output_32frame_exp