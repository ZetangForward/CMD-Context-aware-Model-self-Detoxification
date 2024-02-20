export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
FILE_PATH="../dataset/corresponding_model/rtp_cmd.json"
SAVED_PATH="../ckp/"

python -m torch.distributed.launch --nproc_per_node 8 train.py \
    --model_name_or_path "gpt2-xl" \
    --data_path $FILE_PATH \
    --bf16 True \
    --output_dir $SAVED_PATH \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 False \
    --load_in_8bit False \
    --save_strategy "epoch" 