import os
from dataclasses import field, dataclass
from typing import Dict, Optional, Any, Union, List
import torch
from transformers import AutoTokenizer,TrainerCallback, TrainingArguments, TrainerState, TrainerControl,HfArgumentParser
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from modeling_gpt2 import GPT2LMHeadCLModel
from transformers import Trainer
from basedataset import BaseData
from peft import (
    LoraConfig,
)
from mapping import get_peft_model
import logging
logging.basicConfig(level=logging.INFO)

@dataclass  
class CustomTrainingArguments(TrainingArguments):  
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training"}) 

@dataclass
class TrainingArguments(CustomTrainingArguments):
    model_name_or_path: Optional[str] = field(default="gpt2-xl")
    data_paths: List[str] = field(default_factory=lambda: ["./train.json"], metadata={"help": "Path to the training data."})
    instruction_length: int = 40
    output_length: int = 512
    save_steps = 60000
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    load_in_8bit: bool = field(default=True)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if torch.distributed.is_initialized():  
            is_main_process = torch.distributed.get_rank() == 0
        else:
            is_main_process = True 

        if is_main_process:

            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")

            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)

        return control

def train():

    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    print(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_seq_length = {
        "max_enc_length": 280,
        "max_dec_length": 400,
    }
    
    tokenizer_args = {
        "truncation": True,
        "padding": "max_length",
        "return_tensors": "pt",
    }

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    
    model = GPT2LMHeadCLModel.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    
    # lora hyperparams
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()


    dataset = BaseData(args.data_paths[0], tokenizer, tokenizer_args, max_seq_length, "train")

    trainer = Trainer(
        model,
        args=args,
        data_collator=dataset.collect_fn,
        train_dataset=dataset,
        callbacks=[SavePeftModelCallback],
    )

    trainer.train()

    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    train()
