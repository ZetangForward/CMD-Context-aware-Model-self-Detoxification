from torch.utils.data import Dataset, DataLoader
import json
from transformers import LlamaTokenizer
import torch

# Template for vanilla alpaca-lora
LLAMA_TEMPLATE_V1 = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    # "prompt_no_input": "\"{instruction}\" ",
    "prompt_no_input": "",
    "response_split": "### Response:"    
}


class BaseData(Dataset):
    
    def __init__(self, file, tokenizer=None, tokenizer_args=None, max_seq_length=None, split="train"):
        with open(file, "r", encoding='utf-8') as f:
            content = json.load(f)
            
        self.content = content
        self.split = split
        self.tokenizer = tokenizer
        self.max_enc_length = max_seq_length["max_enc_length"]
        self.max_dec_length = max_seq_length["max_dec_length"]
        self.tokenizer_args = tokenizer_args
    
    def __getitem__(self, index):
        sample = self.content[index]
        instruction = sample.get("instruction", "")
        input_ = sample.get("input", "")
        output = sample.get("output", "")
        
        # create input text
        if len(input_) != 0:
            sample_ipt = LLAMA_TEMPLATE_V1["prompt_input"].format(instruction=instruction, input=input_)
        else:
            sample_ipt = LLAMA_TEMPLATE_V1["prompt_no_input"].format(instruction=instruction)
        # create label
        label = f"{sample_ipt}{output}"
        prompt_id = label.index("Let's detoxify step by step.")
        sample_ipt = prompt = label[:prompt_id]
        label = label[prompt_id:]
        
        input_terms = self.tokenizer(sample_ipt, max_length=self.max_enc_length, **self.tokenizer_args)
        label_terms = self.tokenizer(label, max_length=self.max_dec_length, **self.tokenizer_args)
        
        label_mask = input_terms["attention_mask"] ^ label_terms["attention_mask"]
        original_label_mask = label_mask
        label_mask = label_mask.bool()
        label_mask = ~label_mask * -100 + original_label_mask
        
        # new_label = torch.where(label_mask == 1, label_terms["input_ids"], -100)
        new_label = torch.where(label_terms["attention_mask"]==1,label_terms["input_ids"], -100)
        # import pdb;pdb.set_trace()
        # return {
        #     "input_ids": input_terms['input_ids'],
        #     "attention_mask": input_terms["attention_mask"],
        #     "labels": label_terms['input_ids'],
        #     # "attention_mask": label_terms["attention_mask"]
        # }
        # import pdb;pdb.set_trace()
        return {
                "input_ids": input_terms['input_ids'],
                "attention_mask": input_terms["attention_mask"],
                "labels": new_label,
                # "attention_mask": label_terms["attention_mask"]
        }
    
    def __len__(self):
        return len(self.content)
    
    @classmethod
    def collect_fn(cls, batch_input):
        # import pdb; pdb.set_trace()
        input_ids = [i["input_ids"] for i in batch_input]
        # src_attention_mask = [i["input"]["attention_mask"] for i in batch_input]
        labels = [i["labels"] for i in batch_input]
        trg_attention_mask = [i["attention_mask"] for i in batch_input]
        
        return {
            "input_ids": torch.cat(input_ids),
            "attention_mask": torch.cat(trg_attention_mask),
            "labels": torch.cat(labels),
            # "trg_attention_mask": torch.cat(trg_attention_mask)
        }
        


    
        
        
    