CUDA_VISIBLE_DEVICES=0 python inference.py \
--save_path {your_save_path} \
--model {your_base_model_path} \
--peft {your_LoRA_or_S-Adapter_path} \
--sys_path {./llama or ./gpt2_xl or ./flan_t5_xl} \ 
--evaluate_file {your_test_file_path}
