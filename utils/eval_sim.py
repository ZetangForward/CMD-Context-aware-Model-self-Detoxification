import torch
import json
import math
from transformers import AutoTokenizer,AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--save", type=str)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class custom_dataset(Dataset):
    def __init__(self, file_path, max_padding_size=128):
        
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens',padding_side='right')
        
        
        # init the max_padding_size
        self.max_seq_len = max_padding_size

        self.dic = {}
        
        # load dataset
        with open(file_path, 'r') as f:
            lst = f.readlines()
            self.contents = [json.loads(i) for i in lst]
            temp = []
        
            for content in self.contents:
                if 'raw_recovered_prompt' in content:
                    prompt = content['raw_recovered_prompt']
                else:
                    prompt = content['prompt']['text']
                generations = content['generation'][len(prompt):]
                self.dic[prompt] = content
                for generation in [generations]:
                    dic = {}
                    dic['prompt'] = prompt
                    dic['generation'] = generation
                    temp.append(copy.deepcopy(dic))
            self.content = temp

            
    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        sample = self.content[index]

        prompt, generation = sample['prompt'], sample['generation']

        tokenier_res = self.tokenizer(prompt, return_tensors="pt", \
            padding="max_length", truncation=True, max_length=self.max_seq_len)
        prompt_text_ids = tokenier_res.input_ids[0]
        attention_mask = tokenier_res.attention_mask[0]

        tokenier_res = self.tokenizer(generation, return_tensors="pt", \
            padding="max_length", truncation=True, max_length=self.max_seq_len)
        generation_text_ids = tokenier_res.input_ids[0]
        generation_attention_mask = tokenier_res.attention_mask[0]
            
        return {
            'prompt': prompt,
            'generation': generation,
            'prompt_ids': prompt_text_ids, 
            'attention_mask': attention_mask, 
            'generation_ids': generation_text_ids,
            'generation_attention_mask': generation_attention_mask
        }
    
    def match(self,sim_score,save_path):
        f = open(save_path,'w')
        for i in range(len(sim_score)):
            self.contents[i]['sim_score'] = sim_score[i]
            f.write(json.dumps(self.contents[i])+'\n')
        




def similarity(dataset,bsz):
    sen_score_lst = []

    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens').to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False)

    for sample in tqdm(dataloader,total=len(dataloader)):    
        
        embeddings1 = {}
        embeddings2 = {}
        embeddings1['input_ids'] = sample['prompt_ids'].to(device)
        embeddings1['attention_mask'] = sample['attention_mask'].to(device)
        embeddings2['input_ids'] = sample['generation_ids'].to(device)
        embeddings2['attention_mask'] = sample['generation_attention_mask'].to(device)

        
        with torch.no_grad():
            e1 = model(**embeddings1)
            e2 = model(**embeddings2)
            e1 = mean_pooling(e1, embeddings1['attention_mask'])
            e2 = mean_pooling(e2, embeddings2['attention_mask'])
        sen_score_lst.extend(torch.cosine_similarity(e1,e2,dim=1).tolist())

    return sen_score_lst

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    dataset = custom_dataset(args.file, max_padding_size=96)
    sim_score = similarity(dataset,4096)
    dataset.match(sim_score,args.save)

main()