from my_funcs import (
    default_transformation, read_json_from_data_dir,
    normalize, embed_query_retrieve_examples, iterate_nearest_dialogs,
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
# from rank_bm25 import BM25Okapi
from refpydst.prompt_formats.python.completion_parser import *
from refpydst.prompt_formats.python.completion_parser import parse_python_modified
from refpydst.evaluate_metrics import evaluate
from vllm import LLM, SamplingParams
import torch
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
import refpydst.prompt_formats.python.demo as python_demo


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
    retrieved_exampels = [e for e in retrieved_exampels if e['ID'] != query_data['ID']][-10:]
    return retrieved_exampels

def make_dict(query_data):
    li = []    
    for generated_example in query_data['generated']:
        tmp = {}
        tmp['dialog'] = {}

        if '[context]' in generated_example:
            contxt_end = generated_example.index('[context]') + len('[context]')
        else:
            contxt_end = 0
    
        if '[utterance_sys]' in generated_example:
            sys_start = generated_example.index('[utterance_sys]')
            sys_end = sys_start + len('[utterance_sys]')
        else:
            sys_start = contxt_end+1
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

        tmp['last_slot_values'] = dict(eval('{'+generated_example[contxt_end:sys_start].strip()+'}'))        
        tmp['dialog']['sys'] = [generated_example[sys_end:usr_start].strip()]
        tmp['dialog']['usr'] = [generated_example[usr_end:belief_start].strip()]

        tmp['turn_slot_values'] = {}
        bs = generated_example[belief_end:].strip() if belief_end is not None else ''
        for slot_val in bs.split(','):
            s_v_list = slot_val.split(':')
            if len(s_v_list) <= 1:
                continue
            if len(s_v_list) > 2:
                s_v_list[1] = ':'.join(s_v_list[1:])
            slot, val = s_v_list[0].strip(), s_v_list[1].strip() 
            try:
                tmp['turn_slot_values'].update(dict(eval('{'+ slot + ':' + val +'}')))
            except:
                continue
        li.append(tmp)
    return li


def get_nl_chat_prompt(data_item, examples, n_examples: int = None,
                        reverse_x_and_y: bool = False, use_null_data_item: bool = False,
                        detailed_state_string: bool = True) -> str:

    msg = [{"role": "system", "content": "You are an expert in Dialogue State Tracking(DST) and python coding.\n"}]
    given_context = data_item['last_slot_values']
    max_n_examples: int = n_examples is not None and n_examples or len(examples)

    # in case for zero-shot learning
    if max_n_examples > 0:
        for example_id, example in enumerate(examples[-max_n_examples:]):
            prefix_msg = f"\n#### Example {example_id + 1} ####\n"
            
            # remove multiple choice in last slot values
            last_slot_values = {s: v.split('|')[0] for s, v in example['last_slot_values'].items()}
            turn_slot_values = {s: v.split('|')[0] for s, v in example['turn_slot_values'].items()}

            last_sys_utt = example['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            
            state_string = 'The dialogue state of the converstion so far is like this: '
            for s, v in last_slot_values.items():
                state_string += f"The value of slot {s.split('-')[1]} of {s.split('-')[0]} is {v}. "
            
            turn_msg = state_string + "\nThe llastest conversation between system and user is like this: "
            if last_sys_utt:
                turn_msg += 'The system said "' + last_sys_utt + '".'
            
            turn_msg += 'The user said "' + example['dialog']['usr'][-1] + '".'
            
            bs_msg = '\nThe dialogue state change due to the lastest turn is like this: '
            for s, v in turn_slot_values.items():
                bs_msg += f"The value of slot {s.split('-')[1]} of {s.split('-')[0]} is {v}. "
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
    
    state_string = 'The dialogue state of the converstion so far is like this: '
    for s, v in last_slot_values.items():
        state_string += f"The value of slot {s.split('-')[1]} of {s.split('-')[0]} is {v}. "
    
    gold_string = '\nThe dialogue state change due to the lastest turn is like this: '
    for s, v in example['turn_slot_values'].items():
        gold_string += f"The value of slot {s.split('-')[1]} of {s.split('-')[0]} is {v}. "

    turn_msg = ''
    if not use_null_data_item:
        turn_msg = state_string + "\nThe lastest conversation between system and user is like this: "
        if last_sys_utt:
            turn_msg += 'The system said "' + last_sys_utt + '".'
        turn_msg += 'The user said "' + example['dialog']['usr'][-1] + '".'
    else:
        pass  # default adds our null input at end
    msg.append({"role": "user","content": prefix_msg+turn_msg})
    # msg.append({"role": "assistant","content": "    agent.state."})
    msg[1]['content'] = 'PriceRange is "dontcare" or "cheap" or "moderate" or "expensive". HotelType is "hotel" or "guest house" or "dontcare". Option is "yes" or "no" or "dontcare". DayOfWeek is "monday" or "tuesday" or "wednesday" or "thursday" or "friday" or "saturday" or "sunday". Area is "dontcare" or "centre" or "east" or "north" or "south" or "west".\
        In the Hotel domain, there are a total of ten slots: name, price_range, type, parking, book_number_of_days, book_day, book_people, area, stars and internet.\ The type of slot "name" is a string, the type of slot "price_range" is PriceRange, the type of slot "type" slot is HotelType, the type of slot "parking" is Option, the type of slot "book_number_of_days" is integer, the type of slot "book_day" is DayOfWeek, the type of slot "book_people" is integer, the type of slot "area" is Area, the type of slot "stars" is integer between 0 and 5 or "dontcare" and the type of slot "internet" is Option. \
        In the Train domain, there are a total of five slots: destination, leave_from, day, book_people, depart_time, and arrive_by_time. The type of slot "destination" is a string, the type of slot "leave_from" is a string, the type of slot "day" is DayOfWeek, the type of slot "book_people" is integer, the type of slot "depart_time" is a string in the format hh:mm or "dontcare", and the type of slot "arrive_by_time" is a string in the format hh:mm or "dontcare".\
        AttractionType is "architecture" or "boat" or "church" or "cinema" or "college" or "concert hall" or "entertainment" or "hotspot" or "multiple sports" or "museum" or "nightclub" or "park" or "special" or "swimming pool" or "theatre" or "dontcare".\
        In the Attraction domain, there are a total of three slots: name, area, and type. The type of slot "name" is a string, the type of slot "area" is Area, and the type of slot "type" is AttractionType.\
        In the Restaurant domain, there are a total of six slots: name, food_type, price_range, area, book_time, and book_day. The type of slot "name" is a string, the type of slot "food_type" is a string, the type of slot "price_range" is PriceRange, the type of slot "area" is Area, the type of slot "book_time" is a string in the format hh:mm or "dontcare", the type of slot "book_day" is DayOfWeek, and the type of slot "book_people" is integer.\
        In the Taxi domain, there are a total of four slots: destination, leave_from, depart_time, and arrive_by_time. The type of slot "destination" is a string, the type of slot "leave_from" is a string, the type of slot "depart_time" is a string in the format hh:mm or "dontcare", and the type of slot "arrive_by_time" is a string in the format hh:mm or "dontcare".'\
        + msg[1]["content"]
    return msg, gold_string


def parse_nl_completion(nl_completion: str, state: Union[MultiWOZDict, ParserBeliefState] = None,
                            exceptions_are_empty: bool = True, **kwargs) -> MultiWOZDict:
    """
    The dialogue state change due to the lastest turn is like this: The value of slot book people of restaurant is 3. 
    The value of slot book time of restaurant is 11:15.
    
     Parses a python completion to a complete dialogue state for the turn, y_t.

    :param python_completion: the dialogue state update in python function-call form
    :param state: the existing dialogue state (y_{t-1})
    :param exceptions_are_empty: If an exception is encountered (un-parseable completion), treat this as containing no
      update to the dialogue state.
    :param kwargs: Not used, but included as other parsers use different arguments.
    :return: the updated dialogue state y_t.
    """
    try:
        full_statement = nl_completion.strip()
        full_statement = full_statement.replace("The dialogue state change due to the lastest turn is like this: ", "")
        
        bs_dict = {}
        for sent in full_statement.split('.'):
            if sent=='':
                continue
            sent = sent.strip()
            s_d_v = sent.split('The value of slot ')[1].split(' of ')
            slot = s_d_v[0]
            domain = s_d_v[1].split('is')[0].strip()
            value = s_d_v[1].split(' is')[1].strip()

            bs_dict[f"{domain}-{slot}"] = value

        return bs_dict
    except Exception as e:
        # print(f"got exception when parsing: {pprint.pformat(e)}")
        # logging.warning(e)
        # if not exceptions_are_empty:
        #     raise e
        return {}


if __name__ == '__main__':

    sample_pool_path = 'data/mw21_100p_train.json'
    query_data_path = 'jun/inference/mw24_test_sample_80.json'
    engine = "/data1/home/haesungpyun/models/Meta-Llama-3-70B-Instruct-GPTQ"
    quantization = 'gptq'


    with open('./mw24_test_sample_80_retgen_result.json', 'r') as f:
        query_dataset = json.load(f)

    print(f"Start evaluating mw24_test_sample_80_retgen ...")
    # Query에 대해 retrieve & generate & evaluate
    total_log = []
    stats = defaultdict(int)
    retrieving_samples = True
    num_retrieved_examples = 100
    n_smapled_examples = 10

    random.seed(42)
    # Randomly select a query data and Retrieve examples from example pool (train data)
    # query_data = query_dataset[random.randint(0, len(dev_dataset))]

    prev_data = None
    
    for query_idx, query_data in enumerate(query_dataset):
        query_data['pred_prior_context'] = prev_data['pred_slot_values'] if prev_data else {}
        modified_data = copy.deepcopy(query_data)
        modified_data['last_slot_values'] = query_data.get('pred_prior_context', {})

        examples_list = []
        completions = [{output: 1} for output in [query_data['og_completion']]+[query_data['completion']]]
        best_completion = [comp.strip().replace('agent.state.', '') for dic in completions for comp,_ in dic.items()]

        predicted_prior_context = query_data.get('pred_prior_context', query_data['last_slot_values'])
        batch_pred_turn_s_v = [parse_nl_completion(comp, predicted_prior_context) for comp in best_completion]

        pred_prev = prev_data.get('pred_slot_values', {}) if prev_data else {}
        
        ###########################################################################

        pred = update_dialogue_state(pred_prev, batch_pred_turn_s_v[0])

        query_data['og_pred_turn_slot_values'] = batch_pred_turn_s_v[0]
        query_data['og_pred_slot_values'] = pred
        
        this_jga, this_acc, this_f1 = evaluate(pred, query_data['slot_values'])
        delta_jga, _, _ = evaluate(batch_pred_turn_s_v[0], query_data['turn_slot_values'])
        
        query_data['og_jga'] = this_jga
        query_data['og_delta_jga'] = delta_jga

        stats['og_jga'] += this_jga
        stats['og_delta_jga'] += delta_jga

        ###########################################################################
        
        pred = update_dialogue_state(pred_prev, batch_pred_turn_s_v[1])

        query_data['pred_turn_slot_values'] = batch_pred_turn_s_v[1]
        query_data['pred_slot_values'] = pred
        
        this_jga, this_acc, this_f1 = evaluate(pred, query_data['slot_values'])
        delta_jga, _, _ = evaluate(batch_pred_turn_s_v[1], query_data['turn_slot_values'])
        
        stats['jga'] += this_jga
        stats['delta_jga'] += delta_jga
        query_data['jga'] = this_jga
        query_data['delta_jga'] = delta_jga

        prev_data = query_data

        print(f"{query_idx+1}/{len(query_dataset)}")
        print('ID-turn-id:', query_data['ID-turn-id'])
        print('The GOLD turn_slot_values:', query_data['turn_slot_values'])
        print('The OG turn_slot_values:', query_data['og_pred_turn_slot_values'])
        print('The Gen EX turn_slot_values:', query_data['pred_turn_slot_values'])
        print()
        print('The GOLD slot_values:', query_data['slot_values'])
        print('The OG slot_values:', query_data['og_pred_slot_values'])
        print('The Gen EX slot_values:', query_data['pred_slot_values'])
        print()
        print(f"The OG JGA in {query_data['ID-turn-id']} : full_jga: {query_data['og_full_jga']}, delta_jga: {query_data['og_delta_jga']}")
        print(f"The Gen Ex JGA in {query_data['ID-turn-id']} : full_jga: {query_data['full_jga']}, delta_jga: {query_data['delta_jga']}")
        print()
        print(f"The current OG JGA: ", stats['og_jga']/(query_idx+1))
        print(f"The current OG Delta JGA: ", stats['og_delta_jga']/(query_idx+1))
        print(f"The current Gen EX JGA: ", stats['jga']/(query_idx+1))
        print(f"The current Gen EX Delta JGA: ", stats['delta_jga']/(query_idx+1))
        print('=====================================================================================================')
        print()
    with open(f'./mw24_test_sample_80_retgen_result.json', 'w') as f:
        json.dump(query_dataset, f, indent=4)
