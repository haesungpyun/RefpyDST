import os
import json 
import copy
import numpy as np
import pandas as pd

from my_funcs import (
    default_transformation, read_json_from_data_dir,
    normalize, embed_query_retrieve_examples,
    make_two_type_msg, get_python_chat_prompt, 
    parse_python_completion, update_dialogue_state, 
    compute_acc, calculate_token_f1, evaluate,
    DataOntologyNormalizer, Ontology,
    copy, defaultdict, random,
    tiktoken, SentenceTransformer
)

import copy

from refpydst.prompt_formats.python.completion_parser import *

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199


with open('data/mw21_0p_train.json', 'r') as f:
    train_data = json.load(f)

normalizer = DataOntologyNormalizer(
        Ontology.create_ontology(),
        # count labels from the train set
        supervised_set=train_data,
        # make use of existing surface form knowledge encoded in ontology.json, released with each dataset
        # see README.json within https://github.com/smartyfh/MultiWOZ2.4/raw/main/data/MULTIWOZ2.4.zip
        counts_from_ontology_file="src/refpydst/db/multiwoz/2.4/ontology.json"
)


def unroll_or(gold, pred):
    for slot, val in gold.items():
            if '|' in val:
                for vv in val.split('|'):
                    if pred.get(slot) == vv:
                        pred[slot] = vv
                        gold[slot] = vv
                        break
    return gold, pred

def sorted_dict(dict_a, by_key=True):
    if by_key:
        return dict(sorted(dict_a.items(), key=lambda item: item[0]))
    else:
        return dict(sorted(dict_a.items(), key=lambda item: item[1], reverse=True))


def categorize_errors(slot, val, tmp, visited, gold, pred, pred_prev, prefix='delta', mode='hall'):
    diff_over = dict(set(pred.items()) - set(gold.items()))
    if mode == 'miss':
        if (val in diff_over.values()):
            pred_slot_val = [(k, v) for k, v in diff_over.items() if v == val]
            s_v_gen = copy.deepcopy(iter(pred_slot_val))
            iters = len(pred_slot_val)
            while iters > 0:
                pred_s, pred_v = next(s_v_gen)
                if (pred_s, pred_v) in gold.items():
                    pred_slot_val.remove((pred_s, pred_v))
                iters -= 1
            if len(pred_slot_val) == 0:
                tmp['error'].append((f'{prefix}miss_total', (slot, val)))
                visited.append((slot, val))
                return tmp, visited 

            else:
                for (confused_slot, v) in pred_slot_val:
                    assert v == val
                    tmp['error'].append((f'{prefix}miss_confuse', (slot, val, confused_slot, v)))
                    visited.append((slot, val))
                    visited.append((confused_slot, v))                         
                return tmp, visited 

        else:
            if val == 'dontcare'and pred.get(slot, None) == None:
                tmp['error'].append((f'{prefix}miss_dontcare', (slot, val)))
                visited.append((slot, val))
                return tmp, visited 
                
            if val == '[DELETE]' and pred.get(slot, None) == None:
                tmp['error'].append((f'{prefix}miss_delete', (slot, val)))
                visited.append((slot, val))
                return tmp, visited 
                
            if slot in pred:
                tmp['error'].append((f'{prefix}hall_val', (slot, val, slot, pred[slot])))
                visited.append((slot, val))
                visited.append((slot, pred[slot]))
                return tmp, visited 

            else:
                if tmp.get('error') is None:
                    print(tmp['error'])
                    print(slot, val)
                tmp['error'].append((f'{prefix}miss_total', (slot, val)))
                visited.append((slot, val))
                return tmp, visited 

    elif mode == 'hall':
        if (slot, val) in diff_over.items():
            if slot in gold:
                try:
                    tmp['error'].append((f'{prefix}hall_val', (slot, gold[slot], slot, val)))
                    visited.append((slot, gold[slot]))
                    visited.append((slot, val))
                    return tmp, visited 

                except:
                    print((slot, gold[slot]))
                    print(slot, val)
                    
            elif slot in pred_prev:
                if val == pred_prev[slot]:
                    # tmp['error'].append((f'{prefix}hall_parrot', (slot, val)))
                    pass
                else:
                    tmp['error'].append((f'{prefix}hall_overwrite', (slot, val)))
                    return tmp, visited 
            else:
                tmp['error'].append((f'{prefix}hall_total', (slot, val)))
                return tmp, visited         
    return tmp, visited


def find_error_case(tmp, prev_log, gold, pred, gold_delta, pred_delta, gold_prev, pred_prev):
    delta_miss = dict(set(gold_delta.items()) - set(pred_delta.items()))
    delta_over = dict(set(pred_delta.items()) - set(gold_delta.items()))

    prev_miss = dict(set(gold_prev.items()) - set(pred_prev.items()))
    prev_over = dict(set(pred_prev.items()) - set(gold_prev.items()))  

    over = dict(set(pred.items()) - set(gold.items()))
    miss = dict(set(gold.items()) - set(pred.items()))
    visited = []

    for err_name, err_s_v in tmp.get('error', []):
        if len(err_s_v) > 2:
            visited.append((err_s_v[-2], err_s_v[-1]))
        visited.append((err_s_v[0], err_s_v[1]))

    for gold_slot, gold_val in delta_miss.items():
        if (gold_slot, gold_val) in visited:
            continue
        tmp, visited = categorize_errors(gold_slot, gold_val, tmp, visited, gold_delta, pred_delta, pred_prev, prefix='delta_', mode='miss')

    for pred_slot, pred_val in delta_over.items():
        if (pred_slot, pred_val) in visited:
            continue
        tmp, visited = categorize_errors(pred_slot, pred_val, tmp, visited, gold_delta, pred_delta,pred_prev, prefix='delta_', mode='hall')
    
    # handle the case which is propagated from the previous turn
    for err_name, err_s_v in prev_log.get('error', []):
        if 'hall' in err_name:
            prev_err_slot, prev_err_val = err_s_v[-2], err_s_v[-1]
        if 'miss' in err_name :
            prev_err_slot, prev_err_val = err_s_v[0], err_s_v[1]

        if (prev_err_slot, prev_err_val) in visited:
            continue
        
        if (prev_err_slot, prev_err_val) in prev_miss.items() or (prev_err_slot, prev_err_val) in prev_over.items():
            if (prev_err_slot, prev_err_val) in over.items() or (prev_err_slot, prev_err_val) in miss.items():
                if 'delete' in err_name:
                    prop_name = 'error_prop_'+'_'.join(err_name.split('_')[-2:])
                    tmp['error'].append((prop_name, err_s_v))
                    visited.append((prev_err_slot, prev_err_val)) 
                if (prev_err_slot, prev_err_val) in delta_miss.items() or (prev_err_slot, prev_err_val) in delta_over.items():
                    continue
                prop_name = 'error_prop_'+'_'.join(err_name.split('_')[-2:])
                tmp['error'].append((prop_name, err_s_v))
                visited.append((prev_err_slot, prev_err_val))
    
    # for gold_slot, gold_val in miss.items():
    #     if (gold_slot, gold_val) in visited:
    #         continue
    #     tmp, visited = categorize_errors(gold_slot, gold_val, tmp, visited, gold, pred, pred_prev, prefix='', mode='miss')

    # for pred_slot, pred_val in over.items():
    #     if (pred_slot, pred_val) in visited:
    #         continue
    #     tmp, visited = categorize_errors(pred_slot, pred_val, tmp, visited, gold, pred, pred_prev, prefix='', mode='hall')
    
    return tmp


with open('outputs/runs/table4/zero_shot/split_v1_train/running_log.json', 'r') as f:
    logs = json.load(f)

f_1_d_1_p_1 = []
f_1_d_1_p_0 = []
f_1_d_0_p_1 = []
f_1_d_0_p_0 = []
f_0_d_1_p_1 = []
f_0_d_1_p_0 = []
f_0_d_0_p_1 = []
f_0_d_0_p_0 = []

n_correct = 0
new_logs = []
prev_log = {}
for idx, data_item in enumerate(logs):
    tmp = {}
    if data_item['turn_id'] == 0:
        prev_log = {}
    
    pred_prev = data_item['pred_prior_context']
    # pred_prev = prev_log.get('pred_slot_values', {})
    gold_prev = data_item['last_slot_values']
    gold_prev, pred_prev = unroll_or(gold_prev, pred_prev)

    pred_delta = iterative_parsing(data_item['completion'], pred_prev)
    pred_delta = normalizer.normalize(pred_delta) if 'DELETE' not in str(pred_delta) else pred_delta
    data_item['iter_parse_pred_delta'] = pred_delta
    gold_delta = data_item['turn_slot_values']
    gold_delta, pred_delta = unroll_or(gold_delta, pred_delta)

    pred = update_dialogue_state(pred_prev, pred_delta)
    gold = data_item['slot_values']
    gold, pred = unroll_or(gold, pred)
    
    tmp['ID'] = data_item['ID']
    tmp['turn_id'] = data_item['turn_id']
    tmp['IDS'] = f"{data_item['ID']}_{data_item['turn_id']}"
    
    if pred==gold:
        n_correct+=1
    
    tmp['rights'] = (int(pred==gold), int(pred_delta==gold_delta), int(pred_prev==gold_prev))
    
    if pred == gold and pred_delta == gold_delta and pred_prev == gold_prev:
        f_1_d_1_p_1.append(data_item)
    elif pred == gold and pred_delta == gold_delta and pred_prev != gold_prev:
        f_1_d_1_p_0.append(data_item)
    elif pred == gold and pred_delta != gold_delta and pred_prev == gold_prev:
        f_1_d_0_p_1.append(data_item)
    elif pred == gold and pred_delta != gold_delta and pred_prev != gold_prev:
        f_1_d_0_p_0.append(data_item)
    elif pred != gold and  pred_delta == gold_delta and pred_prev == gold_prev:
        f_0_d_1_p_1.append(data_item)
    elif pred != gold and pred_delta == gold_delta and pred_prev != gold_prev:
        f_0_d_1_p_0.append(data_item)
    elif pred != gold and pred_delta != gold_delta and pred_prev == gold_prev:
        f_0_d_0_p_1.append(data_item)
    else:
        f_0_d_0_p_0.append(data_item) 
                    
    tmp['slot_values'] = sorted_dict(gold)
    tmp['pred_slot_values'] = sorted_dict(pred)
    tmp['pred_og_slot_values'] = sorted_dict(data_item['pred'])
    tmp['turn_slot_values'] =sorted_dict(gold_delta)
    tmp['pred_turn_slot_values'] = sorted_dict(pred_delta)
    tmp['completion'] = data_item['completion']
    tmp['last_slot_values'] = sorted_dict(gold_prev)
    tmp['pred_last_slot_values'] = sorted_dict(pred_prev)
    
    # tmp['dialog'] = log['dialog']

    tmp['error'] = []
    if tmp['IDS'] == 'MUL2105.json_4':
        print()
    if pred != gold:
        tmp = find_error_case(tmp, prev_log, gold, pred, gold_delta, pred_delta, gold_prev, pred_prev)
        
    tmp = find_error_case(tmp, prev_log, gold, pred, gold_delta, pred_delta, gold_prev, pred_prev)
    
    tmp['error'] = sorted(list(set(tuple(x) for x in tmp['error'])))

    tmp['dialog'] = []
    for sys, user in zip(data_item['dialog']['sys'], data_item['dialog']['usr']):
        tmp['dialog'].append('sys: ' + sys)
        tmp['dialog'].append('usr: ' + user)
        
    new_logs.append(tmp)
    prev_log = tmp

with open('analysis/train_zero_log.json', 'w') as f:
    json.dump(new_logs, f, indent=4)


    name = 'delta_miss_total'
for name in ["delta_miss_confuse", 'delta_miss_total', "delta_miss_delete", "delta_miss_dontcare"\
    "delta_hall_overwrite", "delta_hall_total", "delta_hall_val"]: 
    error_logs = []
    for log in new_logs:
        
        flag = False
        for err in log['error']:
            if name in err[0]:
                flag = True
                break
        if not flag:
            continue
        
        for idx, err in enumerate(log['error']):
            if name not in err[0]:
                continue
            
        error_logs.append(log)

    with open(f'analysis//train_zero_{name}.json', 'w') as f:
        json.dump(error_logs, f, indent=4)