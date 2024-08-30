import json 
from collections import Counter
from collections import defaultdict
from statistics import mean
from tqdm import tqdm

query_data_path = 'jun/inference/beliefonly2_checkpoint.json'
output_path='jun/inference/mw24_test_sample_738_beliefonly2_retgen.json'

def make_dict(query_data):
    li = []    
    for generated_example in query_data['generated']:
        tmp = {}
        tmp['dialog'] = {}

        if '[context]' in generated_example:
            context_end = generated_example.index('[context]') + len('[context]')
        else:
            context_end = 0
    
        if '[utterance_sys]' in generated_example:
            sys_start = generated_example.index('[utterance_sys]')
            sys_end = sys_start + len('[utterance_sys]')
        else:
            sys_start = context_end+1
            sys_end = sys_start
    
        if '[utterance_usr]' in generated_example:
            usr_start = generated_example.index('[utterance_usr]')
            usr_end = usr_start + len('[utterance_usr]')
        else:
            usr_start = sys_end+1
            usr_end = usr_start
        
        if '[belief_state]' in generated_example:
            belief_start = generated_example.index('[belief_state]')
            belief_end = belief_start + len('[belief_state]')
        else:
            belief_start = None
            belief_end = None

        tmp['last_slot_values'] = dict(eval('{'+generated_example[context_end:sys_start]+'}'))        
        tmp['dialog']['sys'] = [generated_example[sys_end:usr_start]]
        tmp['dialog']['usr'] = [generated_example[usr_end:belief_start]]

        tmp['turn_slot_values'] = {}
        bs = generated_example[belief_end:].strip() if belief_end is not None else ''
        for slot_val in bs.split(','):
            s_v_list = slot_val.split(':')
            if len(s_v_list) <= 1:
                continue
            if len(s_v_list) > 2:
                s_v_list[1] = ':'.join(s_v_list[1:])
            slot, val = s_v_list[0].strip(), s_v_list[1].strip()
            # tmp['slot_values'][slot] = val
            try:
                tmp['turn_slot_values'].update(dict(eval('{'+ slot + ':' + val +'}')))
            except:
                continue
        li.append(tmp)
    return li

def make_dict_change_order(query_data):
    li = []    
    for generated_example in query_data['generated']:
        tmp = {}
        tmp['dialog'] = {}

        if '[belief_state]' in generated_example:
            belief_end = generated_example.index('[belief_state]') + len('[belief_state]')
        else:
            belief_end = 0
    
        if '[utterance_sys]' in generated_example:
            sys_start = generated_example.index('[utterance_sys]')
            sys_end = sys_start + len('[utterance_sys]')
        else:
            sys_start = belief_end+1
            sys_end = sys_start
    
        if '[utterance_usr]' in generated_example:
            usr_start = generated_example.index('[utterance_usr]')
            usr_end = usr_start + len('[utterance_usr]')
        else:
            usr_start = sys_end+1
            usr_end = usr_start
        
        if '[context]' in generated_example:
            context_start = generated_example.index('[context]')
            context_end = context_start + len('[context]')
        else:
            context_start = None
            context_end = None

        tmp['turn_slot_values'] = dict(eval('{'+generated_example[belief_end:sys_start]+'}'))        
        tmp['dialog']['sys'] = [generated_example[sys_end:usr_start]]
        tmp['dialog']['usr'] = [generated_example[usr_end:context_start]]

        tmp['last_slot_values'] = {}
        bs = generated_example[context_end:].strip() if belief_end is not None else ''
        for slot_val in bs.split(','):
            s_v_list = slot_val.split(':')
            if len(s_v_list) <= 1:
                continue
            if len(s_v_list) > 2:
                s_v_list[1] = ':'.join(s_v_list[1:])
            slot, val = s_v_list[0].strip(), s_v_list[1].strip()
            # tmp['slot_values'][slot] = val
            try:
                tmp['last_slot_values'].update(dict(eval('{'+ slot + ':' + val +'}')))
            except:
                continue
        li.append(tmp)
    return li

with open(query_data_path, 'r') as f:
    query_dataset = json.load(f)

new_data = []
for query_data in query_dataset:
    try:
        li = make_dict(query_data)
    except:
        li = make_dict(query_data)
    
    query_data['generate_items'] = li

##### rating f1-score #####

def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP) != 0 else 0
        recall = TP / float(TP+FN) if (TP+FN) != 0 else 0
        F1 = 2 * precision * recall / \
            float(precision + recall) if (precision+recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return float(F1)

def multival_to_single(belief):
    return [f"{'-'.join(sv.split('-')[:2])}-{(sv.split('-')[-1]).split('|')[0]}" for sv in belief]


# mean of slot similarity and value similarity
def compute_sv_sim(gold, pred, onescore=False):

    if type(gold) == dict:
        gold = [f"{k}-{v}" for k, v in gold.items()]
    if type(pred) == dict:
        pred = [f"{k}-{v}" for k, v in pred.items()]

    gold = multival_to_single(gold)
    pred = multival_to_single(pred)

    value_sim = compute_prf(gold, pred)

    gold = ['-'.join(g.split('-')[:2]) for g in gold]
    pred = ['-'.join(g.split('-')[:2]) for g in pred]
    slot_sim = compute_prf(gold, pred)

    if onescore:
        return value_sim + slot_sim - 1
    else:
        return value_sim, slot_sim

def evaluate_single_query_ex2(data, ex):

    query_turn_sv = data['turn_slot_values']
    query_sv = data['last_slot_values']

    turn_value_sims = []
    turn_slot_sims = []
    all_value_sims = []
    all_slot_sims = []
    if ex=="gen":
        examples = data['generate_items']
    elif ex =="ret":
        examples = data['retrieve_example']
    elif ex =="rand":
        examples = data['random_example']
    else:
        raise ValueError(f"Invalid value for 'ex': {ex}. Expected 'gen', 'ret', or 'rand'.")
    
    for ex in examples:
        this_turn_sv = ex['turn_slot_values']
        this_sv = ex['last_slot_values']

        turn_value_sim, turn_slot_sim = compute_sv_sim(
            query_turn_sv, this_turn_sv, onescore=False)
        all_value_sim, all_slot_sim = compute_sv_sim(query_sv, this_sv, onescore=False)

        turn_value_sims.append(turn_value_sim)
        turn_slot_sims.append(turn_slot_sim)
        all_value_sims.append(all_value_sim)
        all_slot_sims.append(all_slot_sim)
    
    generated_f1 = [s + v*0.3 for s, v in zip(turn_slot_sims, turn_value_sims)]

    return generated_f1

def evaluate_retriever_on_dataset2(dataset, ex="gen"):

    for i in tqdm(range(len(dataset))):
        dataset[i]['generated_f1'] = evaluate_single_query_ex2(
            dataset[i], ex)

    return dataset

new_dataset = evaluate_retriever_on_dataset2(query_dataset,ex="gen")


with open(output_path, 'w') as f:
    json.dump(new_dataset, f, indent=4)

#####################################################################
def evaluate_single_query_ex(data, ex):

    query_turn_sv = data['turn_slot_values']
    query_sv = data['last_slot_values']

    turn_value_sims = []
    turn_slot_sims = []
    all_value_sims = []
    all_slot_sims = []
    if ex=="gen":
        examples = data['generate_items']
    elif ex =="ret":
        examples = data['retrieve_example']
    elif ex =="rand":
        examples = data['random_example']
    else:
        raise ValueError(f"Invalid value for 'ex': {ex}. Expected 'gen', 'ret', or 'rand'.")
    
    for ex in examples:
        this_turn_sv = ex['turn_slot_values']
        this_sv = ex['last_slot_values']

        turn_value_sim, turn_slot_sim = compute_sv_sim(
            query_turn_sv, this_turn_sv, onescore=False)
        all_value_sim, all_slot_sim = compute_sv_sim(query_sv, this_sv, onescore=False)

        turn_value_sims.append(turn_value_sim)
        turn_slot_sims.append(turn_slot_sim)
        all_value_sims.append(all_value_sim)
        all_slot_sims.append(all_slot_sim)
    
    return mean(turn_value_sims), mean(turn_slot_sims), mean(all_value_sims), mean(all_slot_sims)

def evaluate_retriever_on_dataset(dataset, ex="gen"):
    turn_value_sims = []
    turn_slot_sims = []
    all_value_sims = []
    all_slot_sims = []

    for ds in tqdm(dataset):
        turn_value_sim, turn_slot_sim, all_value_sim, all_slot_sim = evaluate_single_query_ex(
            ds, ex)
        turn_value_sims.append(turn_value_sim)
        turn_slot_sims.append(turn_slot_sim)
        all_value_sims.append(all_value_sim)
        all_slot_sims.append(all_slot_sim)

    return round(mean(turn_value_sims), 5), round(mean(turn_slot_sims), 5), round(mean(all_value_sims), 5), round(mean(all_slot_sims), 5)

print("----------------------------------------------------")
print("generated examples f1-score: ",evaluate_retriever_on_dataset(new_dataset, ex="gen"))