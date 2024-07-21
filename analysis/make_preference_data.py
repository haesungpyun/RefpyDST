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

from my_openai_key import get_openai_key
openai.api_key = get_openai_key()

from refpydst.utils.general import read_json, read_json_from_data_dir, get_output_dir_full_path
from refpydst.retriever.code.mixed_retriever import MixedRetriever
from refpydst.retriever.decoders.sampling_topk import SamplingTopK

with open('./data/mw21_5p_train_v1.json', 'r') as f:
    train_dataset = json.load(f)

path = './outputs/runs/table4/5p/smapling_exp/split_v1_topk_bm_5_fs_5_0523_0315/running_log.json'
with open(path, 'r') as f:
    logs = json.load(f)

len(logs)

retriever_full_path: str = get_output_dir_full_path("./outputs/runs/retriever/mw21_5p_train/referred_states/split_v1/")
retriever = MixedRetriever(**{
    "model_path": retriever_full_path,
    "search_index_filename": os.path.join(retriever_full_path, "train_index.npy"),
    **{
        "datasets": [train_dataset],
        "sampling_method": "pre_assigned",
        **({"state_transformation": "ref_aware"} or {})
    }})


def sv_dict_to_nl(slot_values):
    return ', '.join([f"{k.split('-')[1]} of {k.split('-')[0]} is {v}" for k, v in slot_values.items()])

train_data_ids = [data['ID']+'_turn_'+str(data['turn_id']) for data in train_dataset]

new_data = defaultdict(list)

for data_item in logs:
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
        # question_str = '[context] ' + sv_dict_to_nl(modified_item['last_slot_values'])
        question_str += ' [utterance_sys] ' + str(modified_item['dialog']['sys'][-1])
        question_str += ' [utterance_usr] ' + str(modified_item['dialog']['usr'][-1])
        new_data['question'].append(question_str)

        chosen_str = '[context] ' + str(chosen_item['last_slot_values'])
        # chosen_str = '[context] ' + sv_dict_to_nl(chosen_item['last_slot_values'])
        chosen_str += ' [utterance_sys] ' + str(chosen_item['dialog']['sys'][-1])
        chosen_str += ' [utterance_usr] ' + str(chosen_item['dialog']['usr'][-1])
        chosen_str += ' [belief_state] ' + str(chosen_item['slot_values'])
        # chosen_str += '[belief_state] ' + sv_dict_to_nl(chosen_item['slot_values'])
        new_data['chosen'].append(chosen_str)
        
        rejected_str = '[context] ' + str(rejected_item['last_slot_values'])
        # rejected_str = '[context] ' + sv_dict_to_nl(rejected_item['last_slot_values'])
        rejected_str += ' [utterance_sys] ' + str(rejected_item['dialog']['sys'][-1])
        rejected_str += ' [utterance_usr] ' + str(rejected_item['dialog']['usr'][-1])
        rejected_str += ' [belief_state] ' + str(rejected_item['slot_values'])
        # rejected_str += '[belief_state] ' + sv_dict_to_nl(rejected_item['slot_values'])
        new_data['rejected'].append(rejected_str)
    break

with open('./analysis/preference_data.json', 'w') as f:
    json.dump(new_data, f, indent=4)