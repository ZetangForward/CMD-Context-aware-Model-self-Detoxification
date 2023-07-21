import os
from dataclasses import field, dataclass
from typing import Dict, Optional, Any, Union
import sys
from accelerate import Accelerator
from torch import nn
import torch
from transformers import TrainingArguments  
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM,AutoModelForCausalLM,AutoTokenizer
from transformers import Trainer
from basedataset import BaseData
# from peft import (
#     LoraConfig,
#     get_peft_model_state_dict,
#     prepare_model_for_int8_training,
#     set_peft_model_state_dict,
# )

from bottleneck_re import BottleneckConfig
from mapping2 import get_peft_model
from typing import List
import logging
logging.basicConfig(level=logging.INFO)

@dataclass  
class CustomTrainingArguments(TrainingArguments):  
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training"}) 

@dataclass
class TrainingArguments(CustomTrainingArguments):
    model_name_or_path: Optional[str] = field(default="google/flan-t5-base")
    data_paths: List[str] = field(default_factory=lambda: ["./alpaca_data.json"], metadata={"help": "Path to the training data."})
    instruction_length: int = 40
    output_length: int = 160
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    load_in_8bit: bool = field(default=True)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules = [
        "q_proj",
        "v_proj",
    ],

def train():

    accelerator = Accelerator()
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # tokenizer.pad_token_id = (
    #     0  # unk. we want this to be different from the eos token
    # )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.unk_token_id = 0

    # tokenizer.bos_token_id = 1
    # tokenizer.eos_token_id = 2

    # tokenizer.padding_side = "right"  ## Allow batched inference
    max_seq_length = {
        "max_enc_length": 512,   
        "max_dec_length": 512,   
    }
    
    tokenizer_args = {
        "truncation": True,
        "padding": "max_length",
        "return_tensors": "pt",
    }

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        # gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    
    # lora  hyperparams
    if 'lora' in args.output_dir:
        from peft import get_peft_model,LoraConfig
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj","v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # bottleneck hyperparams
    else:
        config = BottleneckConfig(
                bottleneck_size=256,
                non_linearity='tanh',
                adapter_dropout=0.0,
                use_parallel_adapter=False,
                use_adapterp=False,
                target_modules=None,
                scaling=1.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
    
    model = get_peft_model(model, config)
    # import pdb;pdb.set_trace()
    
    model.print_trainable_parameters()
    # import pdb;pdb.set_trace()
    dataset = BaseData(args.data_paths[0], tokenizer, tokenizer_args, max_seq_length, "train")
    
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collect_fn)
    trainer = Trainer(
        model,
        args=args,
        data_collator=dataset.collect_fn,
        train_dataset=dataset,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    train()
