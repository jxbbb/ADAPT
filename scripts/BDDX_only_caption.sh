CUDA_VISIBLE_DEVICES=4,5,6,7 \
NCCL_P2P_DISABLE=1 \
OMPI_COMM_WORLD_SIZE="4" \
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=45978 src/tasks/run_adapt.py \
        --config src/configs/VidSwinBert/BDDX_two_default.json \
        --train_yaml BDDX/training_32frames.yaml \
        --val_yaml BDDX/testing_32frames.yaml \
        --per_gpu_train_batch_size 12 \
        --per_gpu_eval_batch_size 24 \
        --num_train_epochs 40 \
        --learning_rate 0.0004 \
        --max_num_frames 4 \
        --pretrained_2d 0 \
        --backbone_coef_lr 0.05 \
        --mask_prob 0.5 \
        --max_masked_token 45 \
        --zero_opt_stage 1 \
        --mixed_precision_method deepspeed \
        --deepspeed_fp16 \
        --gradient_accumulation_steps 1 \
        --learn_mask_enabled \
        --loss_sparse_w 0.5 \
        --use_sep_cap \
        --output_dir ./output/base_without_multitask
