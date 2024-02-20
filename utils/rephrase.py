from transformers import pipeline
import json
from datasets import load_dataset
import torch
from transformers import pipeline,AutoTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--file",type=str)
parser.add_argument("--save",type=str)
args = parser.parse_args()

f = open(args.file,'r')
f_w = open(args.save,'w')

lst = f.readlines()

tokenizer = AutoTokenizer.from_pretrained('/nvme/hf_models/genius-large')
genius = pipeline("text2text-generation", model='/nvme/hf_models/genius-large', device=0)

for i in tqdm(range(len(lst)),total=len(lst)):
    lst[i] = json.loads(lst[i])
    if lst[i]['prompt']['toxicity'] > 0.5:
        length = len(tokenizer.tokenize(lst[i]['mask_prompt']))
        mask_prompt = lst[i]['mask_prompt'].replace('<MASK>','<mask>')
        mask_num = mask_prompt.count('<mask>')
        generate = genius(mask_prompt, top_p=0.9, temperature=1.0,do_sample=True, max_length=length+5*mask_num)[0]['generated_text']
        lst[i]['raw_recovered_prompt'] = generate
    f_w.write(json.dumps(lst[i])+'\n')
    f_w.flush()


