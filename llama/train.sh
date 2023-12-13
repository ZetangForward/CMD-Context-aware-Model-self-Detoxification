python fastchat.py \
###
 # @Author: ZetangForward 1
 # @Date: 2023-12-12 03:45:26
 # @LastEditors: ZetangForward 1
 # @LastEditTime: 2023-12-13 08:14:30
 # @FilePath: /Detox-CoT/llama/train.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
    --model_name_or_path /zecheng/model_hub/Llama-2-7b-hf \
    --data_path /workspace/zecheng/Detox-CoT/dataset/download/Detox-Chain/llama_7b.json \
    --output_dir /zecheng/detox-cot/llama2 \
    --num_train_epochs 8 \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --load_in_8bit False

