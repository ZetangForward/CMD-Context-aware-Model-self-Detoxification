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
    dic = {}
    if 'mask_prompt' in lst[i]:
        dic['label'] = 'toxic'
        dic['instruction'] = ''
        if lst[i]["sim_score"] > 0.4: #toxic cont
            toxic_prompt_cont = f"###{lst[i]['prompt']['text']} ###Let's detoxify step by step. ###The text is toxic. ###Mask the toxic portion of the text: ###{lst[i]['mask_prompt']} ###Find one or several non-toxic and appropriate tokens to replace <MASK> and the masked text can be rephrased into a non-toxic version: ###{lst[i]['raw_recovered_prompt']} ###The rephrased prompt has enough context. ###The Continuation is: ###{lst[i]['generation']}###"
            dic['output'] = toxic_prompt_cont
        else:
            toxic_prompt_no_cont = f"###{lst[i]['prompt']['text']} ###Let's detoxify step by step. ###The text is toxic. ###Mask the toxic portion of the text: ###{lst[i]['mask_prompt']} ###Find one or several non-toxic and appropriate tokens to replace <MASK> and the masked text can be rephrased into a non-toxic version: ###{lst[i]['raw_recovered_prompt']} ###The rephrased prompt does not have enough context, so there is no need to generate a continuation."
            dic['output'] = toxic_prompt_no_cont
        cot_lst.append(dic)
    else:
        dic['label'] = 'non-toxic'
        if lst[i]["sim_score"] > 0.4:
            non_toxic_prompt_cont = f"###{lst[i]['prompt']['text']} ###Let's detoxify step by step. ###The text is non-toxic. ###It has enough context. ###The Continuation is: ###{lst[i]['generation']}###"
            dic['output'] = non_toxic_prompt_cont
        else:
            non_toxic_prompt_no_cont = f"###{lst[i]['prompt']['text']} ###Let's detoxify step by step. ###The text is non-toxic. ###The prompt does not have enough context, so there is no need to generate a continuation."
            dic['output'] = non_toxic_prompt_no_cont
        cot_lst.append(dic)



f_w.write(json.dumps(cot_lst))