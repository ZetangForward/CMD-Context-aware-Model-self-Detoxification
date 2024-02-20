import json
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

f = open(args.input,'r')
f_w = open(args.output,'w')

lst = f.readlines()

cot_lst = []


for i in range(len(lst)):
    lst[i] = json.loads(lst[i])
    combined = sorted(zip(lst[i]['cont_toxicity'],lst[i]['generation']))
    lst[i]['generation'] = [item[1] for item in combined]
    lst[i]['cont_toxicity'] = [item[0] for item in combined]

    lst[i]['generation'] = lst[i]['generation'][:1] + lst[i]['generation'][-3:]
    lst[i]['cont_toxicity'] = lst[i]['cont_toxicity'][:1] + lst[i]['cont_toxicity'][-3:]
    dic = {}
    if 'mask_prompt' in lst[i]:
        if lst[i]['cont_toxicity'][0] > 0.5:
            continue
        dic['label'] = 'toxic'
        dic['instruction'] = ''
        toxic_prompt_cont = f"# Text: {lst[i]['prompt']['text']} # Let's detoxify step by step. # The context is toxic. # Detect the toxic segment within the context: {lst[i]['mask_prompt']} # Detoxify the toxic segments: {lst[i]['recover_prompt']} # Generation: {lst[i]['generation'][0]} #"
        toxic_prompt_cont_cl_template = f"# Text: {lst[i]['prompt']['text']} # Let's detoxify step by step. # The context is toxic. # Detect the toxic segment within the context: {lst[i]['mask_prompt']} # Detoxify the toxic segments: {lst[i]['recover_prompt']} # Generation:"
        dic['output'] = toxic_prompt_cont
        dic['cont_generation'] = [lst[i]['generation'][0]]
        for idx in range(1,len(lst[i]['generation'])):
            if lst[i]['cont_toxicity'][idx] > 0.3:
                dic['cont_generation'].append(lst[i]['generation'][idx])
        dic['cl_template'] = toxic_prompt_cont_cl_template
        dic['toxicity'] = lst[i]['toxicity']
        cot_lst.append(dic)
    else:
        if lst[i]['cont_toxicity'][0] > 0.5:
            continue
        dic['label'] = 'non-toxic'
        non_toxic_prompt_cont = f"# Text: {lst[i]['prompt']['text']} # Let's detoxify step by step. # The context is non-toxic. # Generation: {lst[i]['generation'][0]} #"
        non_toxic_prompt_cont_cl_template = f"# Text: {lst[i]['prompt']['text']} # Let's detoxify step by step. # The context is non-toxic. # Generation:"
        dic['output'] = non_toxic_prompt_cont
        dic['cl_template'] = non_toxic_prompt_cont_cl_template
        dic['cont_generation'] = [lst[i]['generation'][0]]
        for idx in range(1,len(lst[i]['generation'])):
            if lst[i]['cont_toxicity'][idx] > 0.3:
                dic['cont_generation'].append(lst[i]['generation'][idx])
        dic['toxicity'] = lst[i]['toxicity']
        cot_lst.append(dic)



f_w.write(json.dumps(cot_lst))