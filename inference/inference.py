import torch
import pdb
import json
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration,AutoModelForCausalLM,AutoConfig,LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import Dataset, DataLoader
import transformers
import argparse



parser = argparse.ArgumentParser(description="inference")
parser.add_argument("--bsz", type=int,default=1)
parser.add_argument("--save_path", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--peft", type=str)
parser.add_argument("--sys_path", type=str)
parser.add_argument("--evaluate_file", type=str)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


LOCAL_MODEL_PATH = args.model
FILE_PATH =  args.evaluate_file
PEFT_WEIGHT = args.peft
SAVE_PATH = args.save_path

class custom_dataset(Dataset):
    def __init__(self, file_path, max_padding_size=85, max_gen_len=128):
        
        # load tokenizer
        if 'llama' in args.model or 'alpaca' in args.model:
            self.tokenizer = LlamaTokenizer.from_pretrained(LOCAL_MODEL_PATH,padding_side='left')
            self.tokenizer.pad_token_id = 0
        if 'gpt' in args.model:
            self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH,padding_side='left')
            self.tokenizer.pad_token = self.tokenizer.eos_token  # gpt2 tokenizer has no pad_token
        if 't5' in args.model:
            self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH,padding_side='left')
        
        # init the max_padding_size
        self.max_seq_len = max_padding_size
        self.max_gen_len = max_gen_len
        
        # load dataset
        with open(file_path, 'r') as f:
            lst = f.readlines()
            print('save path:',SAVE_PATH)
            self.content = [json.loads(i) for i in lst]

            
    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        sample = self.content[index]
        prompt = sample['prompt']['text']
        prompt = '###' + prompt + ' ###'
        prompt_text = prompt
        
        
        tokenier_res = self.tokenizer(prompt_text, return_tensors="pt", \
            padding="max_length", truncation=True, max_length=self.max_seq_len)
        prompt_text_ids = tokenier_res.input_ids[0]
        attention_mask = tokenier_res.attention_mask[0]
        
    
        
        return {
            'prompt': prompt,
            'prompt_text_ids': prompt_text_ids, 
            'attention_mask': attention_mask, 
        }
        

def main(file_path):
    
    # define batch size and generation length in inference
    bsz = args.bsz
    max_gen_len = 300
    max_gen_times = 25  # define the max generation times
    
    # load dataset and model
    dataset = custom_dataset(file_path, max_gen_len=max_gen_len)

    if 'llama' in args.model or 'alpaca' in args.model:
        model = LlamaForCausalLM.from_pretrained(LOCAL_MODEL_PATH,torch_dtype=torch.float16,device_map='auto')
    if 'gpt' in args.model:
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH,torch_dtype=torch.float16,device_map='auto')
    if 't5' in args.model:
        model = T5ForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH,torch_dtype=torch.float16,device_map='auto')
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
    
    # can only support serial read
    # model = model.to(device)
    model = model.eval()
    
    # open the written file
    f = open(SAVE_PATH, 'w')
    
    # can only support serial reading
    with tqdm(total=len(dataloader)) as pbar:
        for sample in dataloader:
            input_ids = sample['prompt_text_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device)
            prompt = sample['prompt']
            
            if max_gen_times > 1:
                
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
                    pbar.update(1)
                        
                    for ii in range(bsz):
                        candidate = json.dumps(current_res[ii])
                        f.write(candidate + '\n')
                        f.flush()
                
            else:  # simply generate once
                if len(input_ids) < bsz:
                    bsz = len(input_ids)
                generation_res = model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen_len,
                    do_sample=True, 
                    top_p=0.9, 
                    temperature=1.0,
                    return_dict_in_generate=True, 
                    output_scores=True
                )
                
                input_texts = dataset.tokenizer.batch_decode(input_ids,  skip_special_tokens=True)
                generation_texts = dataset.tokenizer.batch_decode(generation_res.sequences, skip_special_tokens=True)

                current_res = [{
                        "origin_prompt": prompt[i],
                        "prompt": input_texts[i],
                        "generation": generation_texts[i],
                    } for i in range(bsz)]
                
                for i in range(bsz):
                    candidate = json.dumps(current_res[i])
                    f.write(candidate + '\n')
                    f.flush()

                pbar.update(1)
        
    f.close()

if __name__ == "__main__":
    
    main(FILE_PATH, SAVE_PATH)
