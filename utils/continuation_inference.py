import os
import torch
import pdb
import json
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration,AutoModelForCausalLM,AutoConfig,LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import Dataset, DataLoader
import transformers
import argparse



parser = argparse.ArgumentParser(description="inference")
parser.add_argument("--save_path", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--peft", type=str)
parser.add_argument("--sys_path", type=str)
parser.add_argument("--file", type=str)
parser.add_argument("--bsz", type=int)
parser.add_argument("--max_new_tokens", type=int)
parser.add_argument("--gen_times", type=int)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

LOCAL_MODEL_PATH = args.model
FILE_PATH =  args.file
PEFT_WEIGHT = args.peft
SAVE_PATH = args.save_path


class custom_dataset(Dataset):
    def __init__(self, file_path, max_padding_size=128, max_gen_len=128):
        
        # load tokenizer
        if 'llama' in args.model.lower() or 'alpaca' in args.model or 'vicuna' in args.model:
            self.tokenizer = LlamaTokenizer.from_pretrained(LOCAL_MODEL_PATH,padding_side='left')
            self.tokenizer.pad_token_id = 0
        if 'gpt' in args.model or 'sgeat' in args.model:
            self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH,padding_side='left')
            self.tokenizer.pad_token = self.tokenizer.eos_token  # gpt2 tokenizer has no pad_token
        if 't5' in args.model:
            self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH,padding_side='left')
        
        # init the max_padding_size
        self.max_seq_len = max_padding_size
        self.max_gen_len = max_gen_len

        self.dic = {}
        
        # load dataset
        with open(file_path, 'r') as f:
            lst = f.readlines()
            print('save path:',SAVE_PATH)
            self.content = []
            for i in lst:
                line = json.loads(i)
                self.content.append(line)
                if 'raw_recovered_prompt' in line:
                    self.dic[line['raw_recovered_prompt']] = line
                else:
                    self.dic[line['prompt']['text']] = line

            
    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        sample = self.content[index]

        if 'raw_recovered_prompt' in sample.keys():
            prompt, label = sample['raw_recovered_prompt'], sample['continuation']['text']
        else:
            prompt, label = sample['prompt']['text'], sample['continuation']['text']
        
        
        tokenier_res = self.tokenizer(prompt, return_tensors="pt", \
            padding="max_length", truncation=True, max_length=self.max_seq_len)
        prompt_text_ids = tokenier_res.input_ids[0]
        attention_mask = tokenier_res.attention_mask[0]
    
        
        return {'prompt': prompt, 'prompt_text_ids': prompt_text_ids,'attention_mask': attention_mask}
    
    def match(self,result,f):
        for i in range(len(result)):
            temp = self.dic[result[i]['prompt']]
            temp['generation'] = result[i]['generation']
            f.write(json.dumps(temp)+'\n')
            

        

def main(file_path):
    
    # define batch size and generation length in inference
    bsz = args.bsz
    max_gen_len = args.max_new_tokens
    max_gen_times = args.gen_times  # define the max generation times
    
    # load dataset and model
    dataset = custom_dataset(file_path, max_gen_len=max_gen_len)
    if 'llama' in args.model.lower() or 'alpaca' in args.model or 'vicuna' in args.model:
        model = LlamaForCausalLM.from_pretrained(LOCAL_MODEL_PATH,torch_dtype=torch.bfloat16,device_map='auto')
    if 'gpt' in args.model or 'sgeat' in args.model:
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH,torch_dtype=torch.float16)
    if 't5' in args.model:
        model = T5ForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH,torch_dtype=torch.float16)
    if args.peft:
        if 'adapter' in args.peft:
            import sys
            sys.path.append(args.sys_path)
            from peft_model import PeftModel
        if 'lora' in args.peft:
            from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.peft,torch_dtype=torch.float16,device_map='auto')
    
    # wrap the dataset with dataloader
    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False)
    
    model = model.eval()
    
    # open the written file
    if not os.path.exists(os.path.dirname(SAVE_PATH)):
        os.makedirs(os.path.dirname(SAVE_PATH))
    f = open(SAVE_PATH, 'w')
    
    with tqdm(total=len(dataloader)) as pbar:
        result = []
        for sample in dataloader:
            input_ids = sample['prompt_text_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device)
            prompt = sample['prompt']
                           
            input_texts = dataset.tokenizer.batch_decode(input_ids,  skip_special_tokens=True)

            if len(input_texts) < bsz:
                bsz = len(input_texts)

            current_res = [{
                    "prompt": prompt[i],
                    "generation": [],
                } for i in range(bsz)]
            
            with torch.no_grad():             
                generation_res = model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen_len,
                    do_sample=True, 
                    top_p=0.9, 
                    temperature=1.0,
                    return_dict_in_generate=True, 
                    output_scores=True,
                    num_return_sequences=max_gen_times
                )
                generation_texts = dataset.tokenizer.batch_decode(generation_res.sequences, 
                                                                skip_special_tokens=True)
                count = 0
                for ii in range(bsz):
                    for jj in range(max_gen_times):
                        current_res[ii]["generation"].append(generation_texts[count])
                        count += 1
                    if len(current_res[ii]["generation"]) == 1:
                        current_res[ii]["generation"] = current_res[ii]["generation"][0]
                pbar.update(1)
                    
                for ii in range(bsz):
                    result.append(current_res[ii])
        dataset.match(result,f)
    f.close()

if __name__ == "__main__":
    
    main(FILE_PATH)
