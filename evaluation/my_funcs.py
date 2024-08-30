import os
import copy
import torch
import random
import numpy as np
from collections import defaultdict
import openai
import tiktoken
from scipy.spatial import KDTree
from sentence_transformers import SentenceTransformer
from typing import Tuple, List, Union, Literal
from sklearn.metrics.pairwise import cosine_similarity
import pprint

from refpydst.utils.general import read_json_from_data_dir
from refpydst.data_types import Turn
import refpydst.prompt_formats.python.demo as python_demo
from refpydst.prompt_formats.python.completion_parser import (
    ParserBeliefState, ParserAgent, parser_belief_state_from_mwoz_dict, 
    replace_state_references, parse_python_completion, iterative_parsing
    )
from refpydst.utils.dialogue_state import update_dialogue_state
from refpydst.evaluate_metrics import compute_acc, compute_prf
from refpydst.prompt_formats.python.completion_parser import parse_python_completion
from refpydst.utils.dialogue_state import update_dialogue_state
from refpydst.evaluate_metrics import evaluate
from refpydst.normalization.data_ontology_normalizer import DataOntologyNormalizer
from refpydst.db.ontology import Ontology
from refpydst.prompt_formats.python.completion_parser import ParserAgent, ParserBeliefState, parser_belief_state_from_mwoz_dict, replace_state_references


def normalize(emb):
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)

def default_transformation(data_item): 
    # use single turn
    context = data_item['last_slot_values']
    sys_utt = data_item['dialog']['sys'][-1]
    usr_utt = data_item['dialog']['usr'][-1]

    history = "[CONTEXT] "
    for k, v in context.items():
        history += f"{' '.join(k.split('-'))}: {v.split('|')[0]}, "
    if sys_utt == 'none':
        sys_utt = ''
    if usr_utt == 'none':
        usr_utt = ''
    history += f" [SYS] {sys_utt} [USER] {usr_utt}"
    return history

def iterate_nearest_dialogs(query_emb, emb_keys, emb_values, k=5):
    query_emb = normalize(query_emb)
    i = 0
    fetch_size: int = k
    while i < len(emb_keys):
        scores, query_result = KDTree(emb_values).query(query_emb, k=fetch_size, p=2)
        if query_result.shape == (1,):
            i += 1
            yield emb_keys[query_result.item()], scores.item()
        else:
            for item, score_item in zip(query_result.squeeze(0)[i:], scores.squeeze(0)[i:]):
                i += 1
                if item.item() >= len(emb_keys):
                    return  # stop iteration!
                yield emb_keys[item.item()], score_item.item()
        fetch_size = min(2 * fetch_size, len(emb_keys))

def get_python_chat_prompt(data_item, examples, n_examples: int = None,
                        reverse_x_and_y: bool = False, use_null_data_item: bool = False,
                        detailed_state_string: bool = True) -> str:

    msg = [{"role": "system", "content": "You are an expert in Dialogue State Tracking(DST) and python coding.\n"}]
    given_context = data_item['last_slot_values']
    max_n_examples: int = n_examples is not None and n_examples or len(examples)

    # in case for zero-shot learning
    if max_n_examples > 0:
        for example_id, example in enumerate(examples[-max_n_examples:]):
            prefix_msg = f"\n    #### Example {example_id + 1} ####\n"
            
            # remove multiple choice in last slot values
            last_slot_values = {s: v.split('|')[0] for s, v in example['last_slot_values'].items()}

            last_sys_utt = example['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            user_string = python_demo.get_user_string(example['dialog']['usr'][-1])
            state_string, update_string = python_demo.get_python_statements(last_slot_values, example['slot_values'],
                                                                            turn_strings=[last_sys_utt, user_string],
                                                                            detailed_state_string=detailed_state_string)
            turn_msg = "    " + state_string + "\n"
            if last_sys_utt:
                turn_msg += "    " + python_demo.get_system_string(last_sys_utt) + "\n"
            turn_msg += "    " + user_string + "\n"
            
            bs_msg = ''
            for s in update_string.split("\n"):
                bs_msg += "    " + s.strip() + "\n"
            if not reverse_x_and_y:
                msg.append({"role": "user","content": prefix_msg+turn_msg})
                msg.append({"role": "assistant","content": bs_msg})
            else:
                msg.append({"role": "user", "content": prefix_msg+bs_msg})
                msg.append({"role": "assistant","content": turn_msg})

    prefix_msg = f"\n    #### Example {max_n_examples + 1} ####\n"
    if given_context is None:
        last_slot_values = {s: v.split('|')[0] for s, v in data_item['last_slot_values'].items()}
    else:
        last_slot_values = given_context
    last_sys_utt = data_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    user_string = python_demo.get_user_string(data_item['dialog']['usr'][-1])
    state_string, _ = python_demo.get_python_statements(last_slot_values, {},
                                                        turn_strings=[last_sys_utt, user_string],
                                                        detailed_state_string=detailed_state_string)
    _, gt_string = python_demo.get_python_statements(last_slot_values, data_item['slot_values'],
                                                        turn_strings=[last_sys_utt, user_string],
                                                        detailed_state_string=detailed_state_string)
    turn_msg = ''
    if not use_null_data_item:
        turn_msg += "    " + state_string + "\n"
        if last_sys_utt:
            turn_msg += "    " + python_demo.get_system_string(last_sys_utt) + "\n"
        turn_msg += "    " + user_string + "\n"
    else:
        pass  # default adds our null input at end
    msg.append({"role": "user","content": prefix_msg+turn_msg})
    # msg.append({"role": "assistant","content": "    agent.state."})
    msg[1]['content'] = 'import abc\nfrom dataclasses import dataclass\nfrom typing import Literal, Union\n\nPriceRange = Literal["dontcare", "cheap", "moderate", "expensive"]\nHotelType = Literal["hotel", "guest house", "dontcare"]\nOption = Literal["yes", "no", "dontcare"]\nDayOfWeek = Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]\nArea = Literal["dontcare", "centre", "east", "north", "south", "west"]\n\n\n@dataclass\nclass Hotel:\n    name: str = None\n    price_range: PriceRange = None\n    type: HotelType = None\n    parking: Option = None\n    book_number_of_days: int = None\n    book_day: DayOfWeek = None\n    book_people: int = None\n    area: Area = None\n    stars: Union[int, Literal["dontcare"]] = None  # between 0 and 5 or dontcare\n    internet: Option = None\n\n\n@dataclass\nclass Train:\n    destination: str = None\n    leave_from: str = None\n    day: DayOfWeek = None\n    book_people: int = None\n    depart_time: str = None  # hh:mm or dontcare\n    arrive_by_time: str = None  # hh:mm or dontcare\n\n\nAttractionType = Literal["architecture", "boat", "church", "cinema", "college", "concert hall", "entertainment",\n                         "hotspot", "multiple sports", "museum", "nightclub", "park", "special", "swimming pool",\n                         "theatre", "dontcare"]\n\n\n@dataclass\nclass Attraction:\n    name: str = None\n    area: Area = None\n    type: AttractionType = None\n\n\n@dataclass\nclass Restaurant:\n    name: str = None\n    food_type: str = None\n    price_range: PriceRange = None\n    area: Area = None\n    book_time: str = None  # hh:mm or dontcare\n    book_day: DayOfWeek = None\n    book_people: int = None\n\n\n@dataclass\nclass Taxi:\n    destination: str = None\n    leave_from: str = None\n    depart_time: str = None  # hh:mm or dontcare\n    arrive_by_time: str = None  # hh:mm or dontcare\n\n\n@dataclass\nclass BeliefState:\n    hotel: Hotel = None\n    train: Train = None\n    attraction: Attraction = None\n    restaurant: Restaurant = None\n    taxi: Taxi = None\n\n\nclass DialogueAgent(abc.ABC):\n\n    state: BeliefState\n\n    @abc.abstractmethod\n    def find_hotel(self, name: str = None, price_range: PriceRange = None, type: HotelType = None,\n                    parking: Option = None, book_number_of_days: int = None, book_day: DayOfWeek = None,\n                    book_people: int = None, area: Area = None, stars: Union[int, Literal["dontcare"]] = None,\n                    internet: Option = None) -> Hotel:\n        pass\n\n    @abc.abstractmethod\n    def find_train(self, destination: str = None, leave_from: str = None, day: DayOfWeek = None,\n                   book_people: int = None, depart_time: str = None, arrive_by_time: str = None) -> Train:\n        pass\n\n    @abc.abstractmethod\n    def find_attraction(self, name: str = None, area: Area = None, type: AttractionType = None) -> Attraction:\n        pass\n\n    @abc.abstractmethod\n    def find_restaurant(self, name: str = None, food_type: str = None, price_range: PriceRange = None,\n                        area: Area = None, book_time: str = None, book_day: DayOfWeek = None,\n                        book_people: int = None, ) -> Restaurant:\n        pass\n\n    @abc.abstractmethod\n    def find_taxi(self, destination: str = None, leave_from: str = None, depart_time: str = None,\n                  arrive_by_time: str = None) -> Taxi:\n        pass\n\n    def get_state(self) -> BeliefState:\n        return self.state\n\n\nif __name__ == \'__main__\':\n    agent = DialogueAgent()\n    state = BeliefState()\n\n' + msg[1]['content']
    return msg, gt_string

def embed_query_retrieve_examples(embedder, example_pool, query_data, emb_keys, emb_values, label_to_idx, num_retrieved_examples=10):
    # Embed Query based on its turn and previous (predicted) slot values 
    with torch.no_grad():
        query_string = default_transformation(query_data)
        query_emb = embedder.encode(query_string, convert_to_numpy=True).reshape(1, -1)
    
    exmple_generator = (
            (example, score)
            for turn_label, score in iterate_nearest_dialogs(query_emb, emb_keys, emb_values, k=5)
                for example in example_pool
                    if example['ID'] == turn_label.split('_')[0] and example['turn_id'] == int(turn_label.split('_')[-1])
        )

    all_considered_examples: List[Tuple[Turn, float]] = \
        [turn_and_score for _, turn_and_score in zip(range(100), exmple_generator)]
    all_embeddings = np.asarray([
        emb_values[label_to_idx[f"{turn['ID']}_turn_{turn['turn_id']}"]]
        for turn, score in all_considered_examples
    ])
    if len(all_considered_examples) == 0:
        raise ValueError("No examples found in the retriever index.")

    result: List[int] = []
    example_scores = np.asarray([1 - 0.5*(score**2) for turn, score in all_considered_examples])
    assert np.all(np.diff(example_scores) <= 0)  # verifies they are decreasing as expected
    while len(result) < num_retrieved_examples:
        best_idx: int = np.argmax(example_scores).item()
        example_scores[best_idx] = -np.inf
        result.append(best_idx)
        best_emb = all_embeddings[best_idx]
        discount = 0.2 * cosine_similarity(best_emb[None, :], all_embeddings).squeeze(0)
        example_scores = example_scores - discount

    retrieved_exampels = [all_considered_examples[i][0] for i in result][::-1]
    retrieved_exampels = [e for e in retrieved_exampels if e['ID'] != query_data['ID']]
    return retrieved_exampels

def make_two_type_msg(msg_chat):
        """Get the python Chat prompt for the query data and retrieved examples
            msg_chat = [{'role':'system', 'content':'system_msg'}, 
            {'role':'user', 'content':'Some Instruction + Example 1 Input'}, 
            {'role':'assistant', 'content':'Example 1 Gold Output'},
            {'role':'user', 'content':'Example 2 Input'}, 
            {'role':'assistant', 'content':'Example 2 Gold Output'},  ...
            {'role':'user', 'content':'Example Target Input'}, 
            {'role':'assistant', 'content':'The answer is:    '}]
            
            msg_chat_usr = [{'role':'system', 'content':'system_msg'}, 
            {'role':'user', 'content':'Some Instruction + Example 1 Input'}, 
            {'role':'assistant', 'content':'Example 1 Gold Output'},
            {'role':'user', 'content':'Example 2 Input'}, 
            {'role':'assistant', 'content':'Example 2 Gold Output'},  ...
            {'role':'user', 'content':'Example Target Input + 'The answer is:    ''}]
            
            msg_one_prompt = [{'role':'system', 'content':'system_msg'}, 
            {'role':'user', 
                'content':'Some Instruction + Example 1 Input + Example 1 Gold Output + \
                        ... + Example Target Input + 'The answer is:    ''}], 
        """

        msg_chat_usr_last = copy.deepcopy(msg_chat)
        msg_chat_usr_last[-2]['content'] += msg_chat_usr_last[-1]['content']
        msg_chat_usr_last.pop()
        msg_one_prompt = copy.deepcopy(msg_chat)
        for _ in range(2, len(msg_one_prompt)):
            msg_one_prompt[1]['content'] += msg_one_prompt[2]['content']
            msg_one_prompt.pop(2)
        
        return msg_chat_usr_last, msg_one_prompt
    
def calculate_token_f1(encoding, gold, pred):
    pred_tokens = encoding.encode(pred)
    gold_tokens = encoding.encode(gold)

    # Token level f1 score
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for token in gold_tokens:
        if token in pred_tokens:
            true_positive += 1
        else:
            false_negative += 1

    for token in gold_tokens:
        if token not in gold_tokens:
            false_positive += 1

    # Calulate f1 avoiding division by zero
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1


