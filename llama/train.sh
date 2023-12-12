python fastchat.py \
    --model_name_or_path /zecheng/model_hub/Llama-2-7b-hf \
    --data_path /workspace/zecheng/Detox-CoT/dataset/download/Detox-Chain/llama_7b.json \
    --output_dir /zecheng/detox-cot/llama2 \
    --num_train_epochs 8 \
    --per_device_train_batch_size 38 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --load_in_8bit False

