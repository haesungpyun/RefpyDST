from my_funcs import (
    default_transformation, read_json_from_data_dir,
    normalize, embed_query_retrieve_examples,
    make_two_type_msg, get_python_chat_prompt, 
    parse_python_completion, update_dialogue_state, 
    compute_acc, calculate_token_f1, evaluate,
    DataOntologyNormalizer, Ontology,
    copy, defaultdict, random,
    openai, tiktoken, SentenceTransformer
)

import os
import json
import numpy as np
from rank_bm25 import BM25Okapi
import pickle
from collections import defaultdict
import copy
import random
from tqdm import tqdm

from my_openai_key import get_openai_key
openai.api_key = get_openai_key()

from refpydst.utils.general import read_json, read_json_from_data_dir, get_output_dir_full_path
from refpydst.retriever.code.mixed_retriever import MixedRetriever
from refpydst.retriever.decoders.sampling_topk import SamplingTopK

checkpoint_path = "/data1/home/haesungpyun/my_refpydst/jun/notebook/full_train_checkpoint.json"
output_file = "/data1/home/haesungpyun/my_refpydst/jun/notebook/full_train_preference_data.json"

# checkpoint 파일이 존재하면 로드
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)
    new_data = checkpoint['new_data']
    start_idx = checkpoint['start_idx']
    print("======checkpoint is loaded======")
    print("start_idx:", start_idx)
    print("new_data length:", len(new_data['question']))
else:
    new_data = {'question': [], 'chosen': [], 'rejected': []}
    start_idx = 0

# 데이터 파일 로드
with open('/home/haesungpyun/my_refpydst/data/mw21_100p_train.json', 'r') as f:
    train_dataset = json.load(f)

path = '/data1/home/haesungpyun/my_refpydst/data/full_log.json'
with open(path, 'r') as f:
    logs = json.load(f)

print('======Loaded the full log!!======')
print("total log:",len(logs))

retriever_full_path: str = get_output_dir_full_path("/home/haesungpyun/my_refpydst/outputs/runs/retriever/mw21_100p_train/referred_states/")

retriever = MixedRetriever(**{
    "model_path": retriever_full_path,
    "search_index_filename": os.path.join(retriever_full_path, "train_index.npy"),
    **{
        "datasets": [train_dataset],
        "sampling_method": "pre_assigned",
        **({"state_transformation": "ref_aware"} or {})
    }})

print('======Hello Start Making Pref Data======')

def sv_dict_to_nl(slot_values):
    return ', '.join([f"{k.split('-')[1]} of {k.split('-')[0]} is {v}" for k, v in slot_values.items()])

train_data_ids = [data['ID']+'_turn_'+str(data['turn_id']) for data in train_dataset]

# 로그 파일을 순회하면서 데이터 처리

for idx, data_item in tqdm(enumerate(logs[start_idx:], start=start_idx), total=len(logs), initial=start_idx):
    retrieved_examples_ids = list(logs[0].get('final_scores', {}).get('score_delta', {}).keys())
    modified_item = copy.deepcopy(data_item)
    modified_item['last_slot_values'] = modified_item.get('pred_prior_context', {})

    if not retrieved_examples_ids:
        query = retriever.data_item_to_embedding(modified_item)
        bm25_query = retriever.data_item_to_bm25_embedding(modified_item)

        sbert_pool=[(turn_label, score) for _, (turn_label, score) in zip(range(50), retriever.retriever.iterate_nearest_dialogs(query, k=50))]    
        bm25_pool = [(turn_label, score) for (turn_label, score) in retriever.retriever.bm25_iterate_nearest_dialogs(bm25_query, k=100)]

        sbert_ids = []
        for turn_label, score in sbert_pool:
            sbert_ids.append(turn_label)

        bm25_pool_filterd = []
        for turn_label, score in bm25_pool:
            if turn_label not in sbert_ids:
                bm25_pool_filterd.append((turn_label, score))
        bm25_pool_filterd = bm25_pool_filterd[:50]

        for sbert_item, bm25_item in zip(sbert_pool, bm25_pool_filterd):
            retrieved_examples_ids.append(sbert_item[0])
            retrieved_examples_ids.append(bm25_item[0])

    negative_pool_ids = list(set(train_data_ids) - set(retrieved_examples_ids))
    for iteration in range(1000):
        chosen_idx = iteration % 100
        chosen_ids = retrieved_examples_ids[chosen_idx]
        if iteration < 300:
            rejected_pool = negative_pool_ids
            rejected_ids = random.choice(rejected_pool)
        elif iteration < 600:
            rejected_pool = retrieved_examples_ids[chosen_idx+1:] + negative_pool_ids
            rejected_ids = random.choice(rejected_pool)
        else:
            random.shuffle(negative_pool_ids)
            rejected_pool =  retrieved_examples_ids[chosen_idx+1:] + negative_pool_ids[:chosen_idx+1]
            rejected_ids = random.choice(rejected_pool)

        chosen_item = train_dataset[train_data_ids.index(chosen_ids)]
        rejected_item = train_dataset[train_data_ids.index(rejected_ids)]

        question_str = '[context] ' + str(modified_item['last_slot_values'])
        question_str += ' [utterance_sys] ' + str(modified_item['dialog']['sys'][-1])
        question_str += ' [utterance_usr] ' + str(modified_item['dialog']['usr'][-1])
        new_data['question'].append(question_str)

        chosen_str = '[context] ' + str(chosen_item['last_slot_values'])
        chosen_str += ' [utterance_sys] ' + str(chosen_item['dialog']['sys'][-1])
        chosen_str += ' [utterance_usr] ' + str(chosen_item['dialog']['usr'][-1])
        chosen_str += ' [belief_state] ' + str(chosen_item['turn_slot_values'])
        new_data['chosen'].append(chosen_str)
        
        rejected_str = '[context] ' + str(rejected_item['last_slot_values'])
        rejected_str += ' [utterance_sys] ' + str(rejected_item['dialog']['sys'][-1])
        rejected_str += ' [utterance_usr] ' + str(rejected_item['dialog']['usr'][-1])
        rejected_str += ' [belief_state] ' + str(rejected_item['turn_slot_values'])
        new_data['rejected'].append(rejected_str)
    
    # 주기적으로 checkpoint 저장
    if idx % 10 == 0:
        with open(checkpoint_path, 'w') as f:
            json.dump({'new_data': new_data, 'start_idx': idx + 1}, f, indent=4)


# 모든 loop 순회 후 최종 데이터 저장
with open(output_file, 'w') as f:
    json.dump(new_data, f, indent=4)

print(f'Final data saved to {output_file}')
