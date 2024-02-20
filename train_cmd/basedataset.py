from torch.utils.data import Dataset, DataLoader
import json
from transformers import LlamaTokenizer
import torch



class BaseData(Dataset):
    
    def __init__(self, file, tokenizer=None, tokenizer_args=None, max_seq_length=None, split="train"):
        with open(file, "r", encoding='utf-8') as f:
            content = json.load(f)
        content = content
        print('file size:',len(content))
        self.content = content
        self.split = split
        self.tokenizer = tokenizer
        self.max_enc_length = max_seq_length["max_enc_length"]
        self.max_dec_length = max_seq_length["max_dec_length"]
        self.tokenizer_args = tokenizer_args
    
    def __getitem__(self, index):

        sample = self.content[index]

        output = sample.get("output", "")

        ind = output.index("# Let's detoxify step by step")
        input = output[:ind]
        output = output[ind:]

        self.tokenizer.padding_side = 'left'
        cl_generation = sample.get('cl_template','')
        cl_generaiton = self.tokenizer(cl_generation,max_length=250,return_tensors='pt',padding="max_length")

        self.tokenizer.padding_side = 'right'
        generation = sample.get('generation','')
        generation = self.tokenizer(generation, max_length=24, **self.tokenizer_args)
        compensate = 4 - generation['input_ids'].shape[0]
        if compensate > 0:
            generation['input_ids'] = torch.cat((generation['input_ids'],torch.full((compensate,24),0)))
            generation['attention_mask'] = torch.cat((generation['attention_mask'],torch.full((compensate,24),0)))

        input_terms = self.tokenizer(input, return_tensors="pt")
        label_terms = self.tokenizer(output, return_tensors="pt")
        
        input_ids = input_terms['input_ids']
        input_attention = input_terms['attention_mask']
        label_ids = label_terms['input_ids']
        label_attention = label_terms['attention_mask']

        input_terms['input_ids'] = torch.cat((input_ids,label_ids,torch.full((1,self.max_enc_length-input_ids.shape[-1]-label_ids.shape[-1]),self.tokenizer.pad_token_id)),dim=1)
        label_terms['input_ids'] = torch.cat((torch.full(input_ids.shape,-100),label_ids,torch.full((1,self.max_enc_length-input_ids.shape[-1]-label_ids.shape[-1]),-100)),dim=1)

        input_terms['attention_mask'] = torch.where(input_terms['input_ids']==self.tokenizer.pad_token_id,0, 1)
        return {
                "input_ids": input_terms['input_ids'],
                "attention_mask": input_terms['attention_mask'],
                "labels": label_terms['input_ids'],
                "generation_input_ids": generation['input_ids'],
                "generation_attention_mask": generation['attention_mask'],
                "cl_input_ids": cl_generaiton['input_ids'],
                'cl_attention_mask': cl_generaiton['attention_mask']
        }
    
    def __len__(self):
        return len(self.content)
    
    @classmethod
    def collect_fn(cls, batch_input):
        input_ids = [i["input_ids"] for i in batch_input]
        attention_mask = [i["attention_mask"] for i in batch_input]
        labels = [i["labels"] for i in batch_input]
        generation_input_ids = [i["generation_input_ids"] for i in batch_input]
        generation_attention_mask = [i["generation_attention_mask"] for i in batch_input]
        cl_input_ids = [i["cl_input_ids"] for i in batch_input]
        cl_attention_mask = [i["cl_attention_mask"] for i in batch_input]

        return {
            "input_ids": torch.cat(input_ids),
            "attention_mask": torch.cat(attention_mask),
            "labels": torch.cat(labels),
            "generation_input_ids": torch.cat(generation_input_ids),
            "generation_attention_mask": torch.cat(generation_attention_mask),
            "cl_input_ids": torch.cat(cl_input_ids),
            'cl_attention_mask': torch.cat(cl_attention_mask)
        }
        


    

    
        
        
    