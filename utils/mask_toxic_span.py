import json
from transformers import AutoTokenizer,AutoConfig
import sys
sys.path.append("/home/amax/zky/CMD/segment_cnn")
from modeling_bert import BertForSpanClassification
import torch
from tqdm import tqdm
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--model_path", type=str)
args = parser.parse_args()


f = open(args.input,'r')
f_w = open(args.output,'w')
lst = f.readlines()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
config = AutoConfig.from_pretrained(args.model_path)
model = BertForSpanClassification.from_pretrained(args.model_path,config=config).to(device)



model.eval()

for i in tqdm(range(len(lst)),total=len(lst)):
    lst[i] = json.loads(lst[i])
    if lst[i]['prompt']['toxicity'] > 0.5:
        text = lst[i]['prompt']['text']
        text = tokenizer(text,return_tensors="pt")
        text['input_ids'] = text['input_ids'].to(device)
        text['token_type_ids'] = text['token_type_ids'].to(device)
        text['attention_mask'] = text['attention_mask'].to(device)
        output,span_score = model(**text)
        span_score = span_score[:,1:-1,:].squeeze(0)
        span_score = torch.argmax(span_score,dim=1)
        lst[i]['tokenize_prompt'] = tokenizer.tokenize(lst[i]['prompt']['text'])
        lst[i]['prompt_span_label'] = span_score.tolist()

        text = tokenizer.encode(lst[i]['prompt']['text'])
        text = text[1:-1]
        span_label = lst[i]["prompt_span_label"]
        assert len(span_label) == (len(text)-1)
        mask_token_id = tokenizer.encode('[MASK]')[1]

        for j in range(len(text)-1):
            if span_label[j] == 1:
                text[j] = mask_token_id
                text[j+1] = mask_token_id
        mask_prompt = tokenizer.decode(text)
        for j in range(len(text),0,-1):
            mask = '[MASK] ' * j + '[MASK]'
            mask_prompt = mask_prompt.replace(mask,'[MASK]')
        mask_prompt = mask_prompt.replace('[MASK]','<MASK>')

        lst[i]['mask_prompt'] = mask_prompt
    f_w.write(json.dumps(lst[i])+'\n')


    