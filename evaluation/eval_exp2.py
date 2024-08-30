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
from tqdm import tqdm
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
    if examples is None:
        examples=[]
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
    for s, v in data_item['turn_slot_values'].items():
        gold_string += f"The value of slot {s.split('-')[1]} of {s.split('-')[0]} is {v}. "

    turn_msg = ''
    if not use_null_data_item:
        turn_msg = state_string + "\nThe lastest conversation between system and user is like this: "
        if last_sys_utt:
            turn_msg += 'The system said "' + last_sys_utt + '".'
        turn_msg += 'The user said "' + data_item['dialog']['usr'][-1] + '".'
    else:
        pass  # default adds our null input at end
    msg.append({"role": "user","content": prefix_msg+turn_msg})
    # msg.append({"role": "assistant","content": "    agent.state."})
    #msg[1]['content'] = 'PriceRange is "dontcare" or "cheap" or "moderate" or "expensive". HotelType is "hotel" or "guest house" or "dontcare". Option is "yes" or "no" or "dontcare". DayOfWeek is "monday" or "tuesday" or "wednesday" or "thursday" or "friday" or "saturday" or "sunday". Area is "dontcare" or "centre" or "east" or "north" or "south" or "west".\
    #    In the Hotel domain, there are a total of ten slots: name, price_range, type, parking, book_number_of_days, book_day, book_people, area, stars and internet.\ The type of slot "name" is a string, the type of slot "price_range" is PriceRange, the type of slot "type" slot is HotelType, the type of slot "parking" is Option, the type of slot "book_number_of_days" is integer, the type of slot "book_day" is DayOfWeek, the type of slot "book_people" is integer, the type of slot "area" is Area, the type of slot "stars" is integer between 0 and 5 or "dontcare" and the type of slot "internet" is Option. \
    #    In the Train domain, there are a total of five slots: destination, leave_from, day, book_people, depart_time, and arrive_by_time. The type of slot "destination" is a string, the type of slot "leave_from" is a string, the type of slot "day" is DayOfWeek, the type of slot "book_people" is integer, the type of slot "depart_time" is a string in the format hh:mm or "dontcare", and the type of slot "arrive_by_time" is a string in the format hh:mm or "dontcare".\
    #    AttractionType is "architecture" or "boat" or "church" or "cinema" or "college" or "concert hall" or "entertainment" or "hotspot" or "multiple sports" or "museum" or "nightclub" or "park" or "special" or "swimming pool" or "theatre" or "dontcare".\
    #    In the Attraction domain, there are a total of three slots: name, area, and type. The type of slot "name" is a string, the type of slot "area" is Area, and the type of slot "type" is AttractionType.\
    #    In the Restaurant domain, there are a total of six slots: name, food_type, price_range, area, book_time, and book_day. The type of slot "name" is a string, the type of slot "food_type" is a string, the type of slot "price_range" is PriceRange, the type of slot "area" is Area, the type of slot "book_time" is a string in the format hh:mm or "dontcare", the type of slot "book_day" is DayOfWeek, and the type of slot "book_people" is integer.\
    #    In the Taxi domain, there are a total of four slots: destination, leave_from, depart_time, and arrive_by_time. The type of slot "destination" is a string, the type of slot "leave_from" is a string, the type of slot "depart_time" is a string in the format hh:mm or "dontcare", and the type of slot "arrive_by_time" is a string in the format hh:mm or "dontcare".'\
    #    + msg[1]["content"]
    msg[1]['content'] = 'PriceRange is "dontcare" or "cheap" or "moderate" or "expensive". HotelType is "hotel" or "guest house" or "dontcare". Option is "yes" or "no" or "dontcare". DayOfWeek is "monday" or "tuesday" or "wednesday" or "thursday" or "friday" or "saturday" or "sunday". Area is "dontcare" or "centre" or "east" or "north" or "south" or "west".\
        In the Hotel domain, there are a total of ten slots: name, pricerange, type, parking, book stay, book day, book people, area, stars and internet.\ The type of slot "name" is a string, the type of slot "pricerange" is PriceRange, the type of slot "type" slot is HotelType, the type of slot "parking" is Option, the type of slot "book stay" is integer, the type of slot "book day" is DayOfWeek, the type of slot "book people" is integer, the type of slot "area" is Area, the type of slot "stars" is integer between 0 and 5 or "dontcare" and the type of slot "internet" is Option. \
        In the Train domain, there are a total of six slots: destination, departure, day, book people, leaveat, and arriveby. The type of slot "destination" is a string, the type of slot "departure" is a string, the type of slot "day" is DayOfWeek, the type of slot "book people" is integer, the type of slot "leaveat" is a string in the format hh:mm or "dontcare", and the type of slot "arriveby" is a string in the format hh:mm or "dontcare".\
        AttractionType is "architecture" or "boat" or "church" or "cinema" or "college" or "concert hall" or "entertainment" or "hotspot" or "multiple sports" or "museum" or "nightclub" or "park" or "special" or "swimming pool" or "theatre" or "dontcare".\
        In the Attraction domain, there are a total of three slots: name, area, and type. The type of slot "name" is a string, the type of slot "area" is Area, and the type of slot "type" is AttractionType.\
        In the Restaurant domain, there are a total of seven slots: name, food, pricerange, area, book time, and book day and book people. The type of slot "name" is a string, the type of slot "food" is a string, the type of slot "pricerange" is PriceRange, the type of slot "area" is Area, the type of slot "book time" is a string in the format hh:mm or "dontcare", the type of slot "book day" is DayOfWeek, and the type of slot "book people" is integer.\
        In the Taxi domain, there are a total of four slots: destination, departure, leaveat, and arriveby. The type of slot "destination" is a string, the type of slot "departure" is a string, the type of slot "leaveat" is a string in the format hh:mm or "dontcare", and the type of slot "arriveby" is a string in the format hh:mm or "dontcare".'\
        + msg[1]["content"]
    return msg, gold_string


def parse_nl_completion(nl_completion: str, state: Union[MultiWOZDict, ParserBeliefState] = None,
                            exceptions_are_empty: bool = True, **kwargs) -> MultiWOZDict:
    try:
        full_statement = nl_completion.strip()
        full_statement = full_statement.replace("The dialogue state change due to the lastest turn is like this: ", "")
        
        bs_dict = {}
        for sent in full_statement.split('.'):
            sent = sent.strip()
            sent = sent.replace("of'", "of ") 
            try:
                s_d_v = sent.split('The value of slot ')[1].split(' of ')
                slot = s_d_v[0].strip().strip("'")
                domain = s_d_v[1].split(' is ')[0].strip().strip("'")
                value = s_d_v[1].split(' is ')[1].strip().strip("'")

                bs_dict[f"{domain}-{slot}"] = value
            except IndexError:
                continue

        return bs_dict
    except Exception as e:
        # print(f"got exception when parsing: {pprint.pformat(e)}")
        # logging.warning(e)
        # if not exceptions_are_empty:
        #     raise e
        return {}

def get_random_examples(data, num_elements=10):
    if len(data) < num_elements:
        raise ValueError("The data list does not contain enough elements to choose from")
    return random.sample(data, num_elements)


if __name__ == '__main__':

    sample_pool_path = '/data1/home/haesungpyun/my_refpydst/data/mw21_100p_train.json'
    query_data_path = '/data1/home/haesungpyun/my_refpydst/jun/inference/mw24_test_sample_738_beliefonly2_retgen.json'
    engine = "/data1/home/haesungpyun/models/Meta-Llama-3-70B-Instruct-GPTQ"
    quantization = 'gptq'


    with open(sample_pool_path, 'r') as f:
        sample_pool = json.load(f)

    # Register all dialogues from the train dataset to example pool and Get all the unique dialogue ids in example pool
    example_pool = []
    selected_dialog_id_from_split = set()
    for dataset in [sample_pool]:
        example_pool += dataset
        selected_dialog_id_from_split.update([dial['ID'] for dial in dataset])

    # Load the all train data index
    retriever_full_path = '/home/haesungpyun/my_refpydst/outputs/runs/retriever/mw21_5p_train/referred_states/split_v1'
    
    search_index_filename = os.path.join(retriever_full_path, "train_index.npy")
    search_embeddings = np.load(search_index_filename, allow_pickle=True).item()    # {'MUL0720.json_turn_10': np.array([0.1, 0.2, ...]), ...}

    # Keep only embeddings for the selected dialogues in split version
    emb_dict = {k: v for k, v in search_embeddings.items() if k.split('_')[0] in selected_dialog_id_from_split}
    emb_keys = list(emb_dict.keys())
    emb_dim = emb_dict[emb_keys[0]].shape[-1]

    # Convert embeddings to array and Normalize them
    emb_values = np.zeros((len(emb_keys), emb_dim))
    for i, k in enumerate(emb_keys):
        emb_values[i] = emb_dict[k]
    emb_values = normalize(emb_values)

    # Create a label to index mapping  {'MUL0720.json_turn_10': 1, ...} 
    label_to_idx = {k: i for i, k in enumerate(emb_keys)}

    # Load the model for embed query
    embedder = SentenceTransformer(retriever_full_path)

    model = LLM(model="/data1/home/haesungpyun/models/Meta-Llama-3-70B-Instruct-GPTQ", quantization='gptq', enforce_eager=False)
    #model = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", quantization='gptq', enforce_eager=False)
    tokenizer = model.get_tokenizer()

    terminators =  [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("")    
    ]

    stop_sequences = ['--', '\n', ';', '#']

    with open(query_data_path, 'r') as f:
        query_dataset = json.load(f)

    print(f"Start evaluating mw24_test_sample_738_retgen ...")
    # Query에 대해 retrieve & generate & evaluate
    total_log = []
    stats = defaultdict(int)
    retrieving_samples = True
    num_retrieved_examples = 100
    n_smapled_examples = 10

    random.seed(42)
    # Randomly select a query data and Retrieve examples from example pool (train data)
    # query_data = query_dataset[random.randint(0, len(dev_dataset))]

    prev_data = {}
    for query_idx, query_data in enumerate(tqdm(query_dataset, desc="Processing Queries")):
        if query_data['turn_id'] == 0:
            prev_data['og_pred_slot_values'] = {}
            prev_data['pred_slot_values'] = {}
            #prev_data['zero_pred_slot_values'] = {}
            prev_data['rand_pred_slot_values'] = {}
            #prev_data['rand1_pred_slot_values'] = {}

        query_data['og_pred_prior_context'] = prev_data['og_pred_slot_values'] if prev_data else {}
        modified_data = copy.deepcopy(query_data)
        modified_data['last_slot_values'] = query_data.get('og_pred_prior_context', {})

        retrieved_examples = embed_query_retrieve_examples(
            embedder, example_pool, modified_data, 
            emb_keys, emb_values, label_to_idx, num_retrieved_examples=num_retrieved_examples)
        msg_chat, gold_python = get_nl_chat_prompt(query_data, retrieved_examples) 

        generated_examples = query_data['generate_items']

        gen_msg_chat, _ = get_nl_chat_prompt(query_data, generated_examples)   
        # msg_chat_usr_last, msg_one_prompt = make_two_type_msg(msg_chat)
        # raise ValueError    

        #zero_msg_chat, _zero = get_nl_chat_prompt(query_data, examples=None)

        random_examples = get_random_examples(sample_pool)
        rand_msg_chat, _rand = get_nl_chat_prompt(query_data, random_examples)

        #random_examples1 = get_random_examples(sample_pool, num_elements=1)
        #rand1_msg_chat, _rand1 = get_nl_chat_prompt(query_data, random_examples1)

        log = defaultdict(dict)
        log['ID-turn-id']= f"{query_data['ID']}-{query_data['turn_id']}"
        log['last_slot_values'] = query_data['last_slot_values'] 
        log['turn_slot_values'] = query_data['turn_slot_values']
        log['slot_values']= query_data['slot_values']
        log['dialog'] = query_data['dialog']
        log['gold-python'] = gold_python

        examples_list = []
        for idx, example in enumerate(retrieved_examples):
            tmp = {}
            tmp['ID_turn-id'] = f"{example['ID']}-{example['turn_id']}"
            tmp['last_slot_values'], tmp['turn_slot_values'], tmp['slot_values'], tmp['sys_dialog'], tmp['usr_dialog'] = \
                example['last_slot_values'], example['turn_slot_values'], example['slot_values'], example['dialog']['sys'], example['dialog']['usr']
            examples_list.append(tmp)
        log["retrieve_example"] = examples_list

        examples_list = []
        for idx, example in enumerate(generated_examples):
            tmp = {}
            tmp['last_slot_values'], tmp['turn_slot_values'], tmp['sys_dialog'], tmp['usr_dialog'] = \
                example['last_slot_values'], example['turn_slot_values'], example['dialog']['sys'], example['dialog']['usr']
            examples_list.append(tmp)
        log["generated_example"] = examples_list

        rand_examples_list = []
        for idx, example in enumerate(random_examples):
            tmp = {}
            tmp['ID_turn-id'] = f"{example['ID']}-{example['turn_id']}"
            tmp['last_slot_values'], tmp['turn_slot_values'], tmp['slot_values'], tmp['sys_dialog'], tmp['usr_dialog'] = \
                example['last_slot_values'], example['turn_slot_values'], example['slot_values'], example['dialog']['sys'], example['dialog']['usr']
            rand_examples_list.append(tmp)
        log["random_example"] = rand_examples_list

        #rand1_examples_list = []
        #for idx, example in enumerate(random_examples1):
        #    tmp = {}
        #    tmp['ID_turn-id'] = f"{example['ID']}-{example['turn_id']}"
        #    tmp['last_slot_values'], tmp['turn_slot_values'], tmp['slot_values'], tmp['sys_dialog'], tmp['usr_dialog'] = \
        #        example['last_slot_values'], example['turn_slot_values'], example['slot_values'], example['dialog']['sys'], example['dialog']['usr']
        #    rand1_examples_list.append(tmp)
        #log["random_example1"] = rand1_examples_list
        #print(rand1_examples_list)
        
        samplig_params = SamplingParams(
            n=1, best_of=1, max_tokens=120, 
            temperature=0, stop=stop_sequences,
            stop_token_ids=terminators)
        msg_chat_ids = tokenizer.apply_chat_template(
            msg_chat, add_generation_prompt=True, return_tensors='pt')

        gen_msg_chat_ids = tokenizer.apply_chat_template(
            gen_msg_chat, add_generation_prompt=True, return_tensors='pt')

        #zero_msg_chat_ids = tokenizer.apply_chat_template(
        #    zero_msg_chat, add_generation_prompt=True, return_tensors='pt')

        rand_msg_chat_ids = tokenizer.apply_chat_template(
            rand_msg_chat, add_generation_prompt=True, return_tensors='pt')

        #rand1_msg_chat_ids = tokenizer.apply_chat_template(
        #    rand1_msg_chat, add_generation_prompt=True, return_tensors='pt')
        
        prompts = [tokenizer.batch_decode(ids, skip_special_tokens=False)[0] for ids in [msg_chat_ids, gen_msg_chat_ids, rand_msg_chat_ids]]
        result = model.generate(prompts, sampling_params=samplig_params)
        
        completions = [{output.outputs[0].text: 1} for output in result]
        best_completion = [comp.strip().replace('agent.state.', '') for dic in completions for comp,_ in dic.items()]
        
        print("best_completion:",best_completion)
        predicted_prior_context = query_data.get('pred_prior_context', query_data['last_slot_values'])
        #batch_pred_s_v = [parse_python_completion(comp, predicted_prior_context) for comp in best_completion]
        #batch_pred_turn_s_v = [parse_state_change(comp, predicted_prior_context) for comp in best_completion]
        batch_pred_turn_s_v = [parse_nl_completion(comp, predicted_prior_context) for comp in best_completion]

        og_pred_prev = prev_data.get('og_pred_slot_values', {}) if prev_data else {}
        pred_prev = prev_data.get('pred_slot_values', {}) if prev_data else {}
        #zero_pred_prev = prev_data.get('zero_pred_slot_values', {}) if prev_data else {}
        rand_pred_prev = prev_data.get('rand_pred_slot_values', {}) if prev_data else {}
        #rand1_pred_prev = prev_data.get('rand1_pred_slot_values', {}) if prev_data else {}

        ###########################################################################
        pred = update_dialogue_state(og_pred_prev, batch_pred_turn_s_v[0])

        query_data['og_completion'] = best_completion[0]
        query_data['og_pred_turn_slot_values'] = batch_pred_turn_s_v[0]
        query_data['og_pred_slot_values'] = pred
        
        log['og_completion'] = best_completion[0]
        log['og_pred_turn_slot_values'] = batch_pred_turn_s_v[0]
        log['og_pred_slot_values'] = pred
        
        this_jga, this_acc, this_f1 = evaluate(pred, query_data['slot_values'])
        delta_jga, _, _ = evaluate(batch_pred_turn_s_v[0], query_data['turn_slot_values'])
        log['og_full_jga'] = this_jga
        log['og_delta_jga'] = delta_jga

        stats['og_jga'] += this_jga
        stats['og_delta_jga'] += delta_jga

        ###########################################################################
        
        pred = update_dialogue_state(pred_prev, batch_pred_turn_s_v[1])

        query_data['completion'] = best_completion[1]
        query_data['pred_turn_slot_values'] = batch_pred_turn_s_v[1]
        query_data['pred_slot_values'] = pred
        
        log['completion'] = best_completion[1]
        log['pred_turn_slot_values'] = batch_pred_turn_s_v[1]
        log['pred_slot_values'] = pred
        
        this_jga, this_acc, this_f1 = evaluate(pred, query_data['slot_values'])
        delta_jga, _, _ = evaluate(batch_pred_turn_s_v[1], query_data['turn_slot_values'])
        log['full_jga'] = this_jga
        log['delta_jga'] = delta_jga


        stats['jga'] += this_jga
        stats['delta_jga'] += delta_jga

        ###########################################################################

        #pred = update_dialogue_state(zero_pred_prev, batch_pred_turn_s_v[2])

        #query_data['zero_completion'] = best_completion[2]
        #query_data['zero_pred_turn_slot_values'] = batch_pred_turn_s_v[2]
        #query_data['zero_pred_slot_values'] = pred
        
        #log['zero_completion'] = best_completion[2]
        #log['zero_pred_turn_slot_values'] = batch_pred_turn_s_v[2]
        #log['zero_pred_slot_values'] = pred
        
        #this_jga, this_acc, this_f1 = evaluate(pred, query_data['slot_values'])
        #delta_jga, _, _ = evaluate(batch_pred_turn_s_v[2], query_data['turn_slot_values'])
        #log['zero_full_jga'] = this_jga
        #log['zero_delta_jga'] = delta_jga

        #stats['zero_jga'] += this_jga
        #stats['zero_delta_jga'] += delta_jga

        ###########################################################################

        pred = update_dialogue_state(rand_pred_prev, batch_pred_turn_s_v[2])

        query_data['rand_completion'] = best_completion[2]
        query_data['rand_pred_turn_slot_values'] = batch_pred_turn_s_v[2]
        query_data['rand_pred_slot_values'] = pred
        
        log['rand_completion'] = best_completion[2]
        log['rand_pred_turn_slot_values'] = batch_pred_turn_s_v[2]
        log['rand_pred_slot_values'] = pred
        
        this_jga, this_acc, this_f1 = evaluate(pred, query_data['slot_values'])
        delta_jga, _, _ = evaluate(batch_pred_turn_s_v[2], query_data['turn_slot_values'])
        log['rand_full_jga'] = this_jga
        log['rand_delta_jga'] = delta_jga

        stats['rand_jga'] += this_jga
        stats['rand_delta_jga'] += delta_jga

        ###########################################################################

        #pred = update_dialogue_state(rand1_pred_prev, batch_pred_turn_s_v[3])

        #query_data['rand1_completion'] = best_completion[3]
        #query_data['rand1_pred_turn_slot_values'] = batch_pred_turn_s_v[3]
        #query_data['rand1_pred_slot_values'] = pred
        
        #log['rand1_completion'] = best_completion[3]
        #log['rand1_pred_turn_slot_values'] = batch_pred_turn_s_v[3]
        #log['rand1_pred_slot_values'] = pred
        
        #this_jga, this_acc, this_f1 = evaluate(pred, query_data['slot_values'])
        #delta_jga, _, _ = evaluate(batch_pred_turn_s_v[3], query_data['turn_slot_values'])
        #log['rand1_full_jga'] = this_jga
        #log['rand1_delta_jga'] = delta_jga

        #stats['rand1_jga'] += this_jga
        #stats['rand1_delta_jga'] += delta_jga

        ###########################################################################
        prev_data = query_data
        total_log.append(log)

        log['og_prompt'] = msg_chat
        log['prompt'] = gen_msg_chat
        #log['zero_prompt'] = zero_msg_chat
        log['rand_prompt'] = rand_msg_chat
        print(f"{query_idx+1}/{len(query_dataset)}")
        print('ID-turn-id:', log['ID-turn-id'])
        print('The GOLD turn_slot_values:', log['turn_slot_values'])
        print('The OG turn_slot_values:', log['og_pred_turn_slot_values'])
        print('The Gen EX turn_slot_values:', log['pred_turn_slot_values'])
        #print('The Zero turn_slot_values:', log['zero_pred_turn_slot_values'])
        print('The Random turn_slot_values:', log['rand_pred_turn_slot_values'])
        #print('The Random 1 turn_slot_values:', log['rand1_pred_turn_slot_values'])
        print()
        print('The GOLD slot_values:', log['slot_values'])
        print('The OG slot_values:', log['og_pred_slot_values'])
        print('The Gen EX slot_values:', log['pred_slot_values'])
        #print('The Zero slot_values:', log['zero_pred_slot_values'])
        print('The Random slot_values:', log['rand_pred_slot_values'])
        #print('The Random 1 slot_values:', log['rand1_pred_slot_values'])
        print()
        print(f"The OG JGA in {log['ID-turn-id']} : full_jga: {log['og_full_jga']}, delta_jga: {log['og_delta_jga']}")
        print(f"The Gen Ex JGA in {log['ID-turn-id']} : full_jga: {log['full_jga']}, delta_jga: {log['delta_jga']}")
        #print(f"The Zero Ex JGA in {log['ID-turn-id']} : full_jga: {log['zero_full_jga']}, delta_jga: {log['zero_delta_jga']}")
        print(f"The Random Ex JGA in {log['ID-turn-id']} : full_jga: {log['rand_full_jga']}, delta_jga: {log['rand_delta_jga']}")
        #print(f"The Random 1 Ex JGA in {log['ID-turn-id']} : full_jga: {log['rand1_full_jga']}, delta_jga: {log['rand1_delta_jga']}")
        print()
        print(f"The current OG JGA: ", stats['og_jga']/(query_idx+1))
        print(f"The current OG Delta JGA: ", stats['og_delta_jga']/(query_idx+1))
        print(f"The current Gen EX JGA: ", stats['jga']/(query_idx+1))
        print(f"The current Gen EX Delta JGA: ", stats['delta_jga']/(query_idx+1))
        #print(f"The current Zero EX JGA: ", stats['zero_jga']/(query_idx+1))
        #print(f"The current Zero EX Delta JGA: ", stats['zero_delta_jga']/(query_idx+1))
        print(f"The current Random EX JGA: ", stats['rand_jga']/(query_idx+1))
        print(f"The current Random EX Delta JGA: ", stats['rand_delta_jga']/(query_idx+1))
        #print(f"The current Random 1 EX JGA: ", stats['rand1_jga']/(query_idx+1))
        #print(f"The current Random 1 EX Delta JGA: ", stats['rand1_delta_jga']/(query_idx+1))
        print('=====================================================================================================')
        print()
    with open(f'jun/eval_exp/mw24_test_sample_738_beliefonly_retgen_result.json', 'w') as f:
        json.dump(total_log, f, indent=4)