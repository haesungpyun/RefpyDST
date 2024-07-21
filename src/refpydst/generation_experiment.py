"""
This class holds details for abstract prompt-based experiments, in which concrete sub-classes would need to define:

1) the language model which generates completions
2) any arguments to that model or experiment details specific to prompting with it

Ultimately, these need to define parse-able dialogue states, the parsing of which may be delegated to
the prompt-format mechanisms
"""
import abc
import copy
import itertools
import json
import logging
import os
import pprint
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Final, Type

import wandb
from refpydst.data_types import Turn, CompletionParser, MultiWOZDict
from tqdm import tqdm

from refpydst.data_types import ExampleListDecoderConfig, ExampleListDecoderType
from refpydst import evaluate_run_log, artifacts
from refpydst.db.ontology import Ontology
from refpydst.error_analysis import slot_level_f1, count_prompts_from_examples
from refpydst.evaluate_metrics import calc_prf, evaluate, compute_prf, compute_acc, calculate_token_f1
from refpydst.normalization.abstract_normalizer import AbstractNormalizer
from refpydst.normalization.data_ontology_normalizer import DataOntologyNormalizer
from refpydst.prompting import PromptGenerator, get_completion_parser
from refpydst.retriever.abstract_example_retriever import ExampleRetriever, MockRetriever
from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder
from refpydst.retriever.code.embed_based_retriever import EmbeddingRetriever
from refpydst.retriever.code.bm25_retriever import BM25Retriever
from refpydst.retriever.code.mixed_retriever import MixedRetriever
from refpydst.retriever.code.error_retriever import ErrorRetriever
from refpydst.retriever.decoders.bm25_max_emb_distance import BM25MaxEmbDistinct
from refpydst.retriever.decoders.mixed_max_emb_distance import MixedMaxEmbDistinct
from refpydst.retriever.decoders.mixed_topk import MixedTopK
from refpydst.retriever.decoders.sampling_topk import SamplingTopK
from refpydst.retriever.decoders.error_topk import ErrorTopK
from refpydst.retriever.decoders.maximize_embedding_distinctness import MaximizeEmbeddingDistinctness
from refpydst.retriever.decoders.top_k import TopKDecoder
from refpydst.retriever.random_retriever import RandomExampleRetriever
import refpydst.prompt_formats.python.demo as python_demo
from refpydst.utils.dialogue_state import group_by_dial_id_and_turn
from refpydst.utils.dialogue_state import update_dialogue_state
from refpydst.utils.general import read_json, read_json_from_data_dir, get_output_dir_full_path
from refpydst.utils.state_recorder import PreviousStateRecorder
from refpydst.wandb_step_logger import WandbStepLogger

_DECODER_CLASSES: Dict[ExampleListDecoderType, Type[AbstractExampleListDecoder]]


class AbstractLMPromptingExperiment(metaclass=abc.ABCMeta):
    """
    Abstract class for managing an experiment which prompts
    """

    test_set: List[Turn]
    use_gold: bool
    prompt_format: str
    only_use_turn_idx: int
    prediction_recorder: PreviousStateRecorder
    prompt_generator: PromptGenerator
    retriever: ExampleRetriever
    num_examples: int
    ontology_file_path: str
    demonstration_mapping: Optional[Dict[str, Dict[str, List[Turn]]]]
    train_set: List[Turn]
    completion_parser: CompletionParser
    mwz_ver: str
    demonstration_decoder: AbstractExampleListDecoder
    output_dir: str
    ontology: Ontology
    normalizer: AbstractNormalizer
    format_example: Optional[Turn]
    logger: WandbStepLogger

    def __init__(self, test_set_path: str, use_gold: bool = False,
                 prompt_format: str = None, turn: int = -1, num_examples: int = 10, retriever_dir: str = None,
                 train_set_path: str = None, demonstration_mapping_path: str = None, mwz_ver: str = "2.4",
                 retriever_type: str = None, decoder_config: Optional[ExampleListDecoderConfig] = None,
                 retriever_args: Dict[str, Any] = None, output_dir: str = None,
                 ontology_file_path: str = "db/multiwoz/2.4/ontology.json",
                 format_example: Optional[Turn] = None, artifact_cache: str = None, **kwargs) -> None:
        self.test_set = read_json_from_data_dir(test_set_path)
        self.use_gold = use_gold
        self.prompt_format = prompt_format
        self.only_use_turn_idx = turn
        self.prediction_recorder = PreviousStateRecorder()  # state recorder
        self.prompt_generator = PromptGenerator()
        self.num_examples = num_examples
        self.train_set = read_json_from_data_dir(train_set_path)
        self.output_dir = output_dir
        self.ontology = Ontology.create_ontology()
        self.ontology_file_path = ontology_file_path
        self.normalizer = DataOntologyNormalizer(
            self.ontology,
            # count labels from the train set
            supervised_set=self.train_set,
            # make use of existing surface form knowledge encoded in ontology.json, released with each dataset
            # see README.json within https://github.com/smartyfh/MultiWOZ2.4/raw/main/data/MULTIWOZ2.4.zip
            counts_from_ontology_file=ontology_file_path
        )
        self.logger = WandbStepLogger()
        # load the selection pool and retriever
        self.retriever = get_retriever_by_type(retriever_type, retriever_dir, retriever_args={
            "datasets": [self.train_set],
            "sampling_method": "pre_assigned",
            **(retriever_args or {})  # default is None, which is not a mapping and throws an exception
        })

        # choose a decoder for selecting example lists, given retrieval scores of individual examples (local parts)
        if not decoder_config:
            self.demonstration_decoder = TopKDecoder()
        
        elif decoder_config.get('decoder_type', 'top_k') == 'top_k':
            self.demonstration_decoder = TopKDecoder()
        
        elif decoder_config['decoder_type'] == 'max_emb_distance':
            if not isinstance(self.retriever, EmbeddingRetriever):
                raise ValueError(f"cannot maximize embedding distance with a retriever of type: {type(self.retriever)}")
            self.demonstration_decoder = MaximizeEmbeddingDistinctness(
                retriever=self.retriever,
                from_n_possible=decoder_config['from_n_possible'],
                discount_factor=decoder_config['discount_factor']
            )
        
        elif decoder_config['decoder_type'] == 'bm25_max_emb_distance':
            if not isinstance(self.retriever, BM25Retriever):
                raise ValueError(f"cannot maximize embedding distance with a retriever of type: {type(self.retriever)}")
            self.demonstration_decoder = BM25MaxEmbDistinct(
                retriever=self.retriever,
                from_n_possible=decoder_config['from_n_possible'],
                discount_factor=decoder_config['discount_factor']
            )
        
        elif decoder_config['decoder_type'] == 'mixed_max_emb_distance':
            if not isinstance(self.retriever, MixedRetriever):
                raise ValueError(f"cannot maximize embedding distance with a retriever of type: {type(self.retriever)}")
            self.demonstration_decoder = MixedMaxEmbDistinct(
                retriever=self.retriever,
                from_n_possible=decoder_config['from_n_possible'],
                discount_factor=decoder_config['discount_factor'],
                operation=decoder_config.get('operation', 'sum'),
                zscore=decoder_config.get('zscore', False)
            )
        
        elif decoder_config['decoder_type'] == 'mixed_top_k':
            if not isinstance(self.retriever, MixedRetriever):
                raise ValueError(f"cannot maximize embedding distance with a retriever of type: {type(self.retriever)}")
            self.demonstration_decoder = MixedTopK(retriever=self.retriever, operation=decoder_config['operation'], zscore=decoder_config['zscore'])
        
        elif decoder_config['decoder_type'] == 'sampling_top_k':
            if not isinstance(self.retriever, MixedRetriever):
                raise ValueError(f"cannot maximize embedding distance with a retriever of type: {type(self.retriever)}")
            self.demonstration_decoder = SamplingTopK(retriever=self.retriever, k=self.num_examples)

        elif decoder_config['decoder_type'] == 'error_top_k':
            if not isinstance(self.retriever, ErrorRetriever):
                raise ValueError(f"cannot maximize embedding distance with a retriever of type: {type(self.retriever)}")
            ground_key = decoder_config.get('ground_key', 'correct')
            self.demonstration_decoder = SamplingTopK(retriever=self.retriever, ground_key=ground_key)
        
        else:
            raise ValueError(f"Unable to construct decoder: {pprint.pformat(decoder_config)}")
        
        self.mwz_ver = mwz_ver
        if demonstration_mapping_path:
            # this may load as a set of tuples in the values, but we should inject the full turn
            self.demonstration_mapping = read_json(demonstration_mapping_path)
            train_set_as_dict = _turn_list_to_dial_id_turn_id_dict(self.train_set)
            for dial_id in self.demonstration_mapping:
                for turn_id in self.demonstration_mapping[dial_id]:
                    self.demonstration_mapping[dial_id][turn_id] = [
                        train_set_as_dict[demo_dial_id][demo_turn_id] for demo_dial_id, demo_turn_id in
                        self.demonstration_mapping[dial_id][turn_id]
                    ]
        else:
            self.demonstration_mapping = None

        self.completion_parser = get_completion_parser(self.prompt_format)
        self.format_example = format_example
        if artifact_cache:
            running_log = artifacts.read_json_artifact(artifact_cache, "running_log.json", alias="latest")
            self.run_log_cache = group_by_dial_id_and_turn(running_log)
        else:
            self.run_log_cache = None

    def get_prompt_text(self, data_item: Turn, examples: List[Turn]) -> str:
        """
        Given a target/inference turn, retrieve self.num_examples demonstrations and build a prompt which includes
        these, according to the format defined when instantiating the Experiment. Return this prompt and the
        demonstrations included in it.
        """
       
        if self.use_gold:
            prompt_text = self.prompt_generator.get_prompt(
                data_item, examples=examples, prompt_format=self.prompt_format)
        else:
            predicted_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)
            prompt_text = self.prompt_generator.get_prompt(
                data_item, examples=examples, given_context=predicted_context, prompt_format=self.prompt_format)
        return prompt_text


    def get_prompt_text_dict(self, data_item: Turn, examples: List[Turn], zero_one_shot:bool=False) -> str:
        """
        Given a target/inference turn, retrieve self.num_examples demonstrations and build a prompt which includes
        these, according to the format defined when instantiating the Experiment. Return this prompt and the
        demonstrations included in it.
        """
        if zero_one_shot:
            ######### NEW CODE #########
            predicted_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)
            prompt_text_dict = defaultdict(str)
            prompt_text_dict["0-shot"] = \
                self.prompt_generator.get_prompt(
                    data_item, examples=[], 
                    given_context=predicted_context, 
                    prompt_format=self.prompt_format,
                    chat_format=True
                )
            for ex in examples:
                prompt_text_dict[f"{ex['ID']}_turn_{ex['turn_id']}"] = self.prompt_generator.get_prompt(
                    data_item, examples=[ex], 
                    given_context=predicted_context, 
                    prompt_format=self.prompt_format,
                    chat_format=True
                )
            if len(examples) > 0:
                prompt_text_dict[f"{len(examples)}-shot"] = \
                    self.prompt_generator.get_prompt(
                        data_item, examples=examples, 
                        given_context=predicted_context, 
                        prompt_format=self.prompt_format,
                        chat_format=True
                    ) 
            return prompt_text_dict
        else:
            if self.use_gold:
                prompt_text = self.prompt_generator.get_prompt(
                    data_item, examples=examples, prompt_format=self.prompt_format)
            else:
                predicted_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)
                prompt_text = self.prompt_generator.get_prompt(
                    data_item, examples=examples, given_context=predicted_context, prompt_format=self.prompt_format, chat_format=True)
            return {f"{len(examples)}-shot":prompt_text}
        
    def get_demonstrations(self, data_item: Turn) -> List[Turn]:
        """
        Given a target/inference turn, retrieve and return self.num_examples demonstrations
        """
        # we've already decided which demonstrations to use, just retrieve these
        examples: List[Turn] = []
        if self.format_example:
            examples.append(self.format_example)
        if self.demonstration_mapping:
            return self.demonstration_mapping[data_item['ID']][str(data_item['turn_id'])]

        num_examples = self.num_examples if isinstance(self.num_examples, int) else sum(self.num_examples.values())
        if self.use_gold:
            examples.extend(self.retriever.item_to_best_examples(data_item, k=self.num_examples,
                                                            decoder=self.demonstration_decoder))
        elif len(examples) < num_examples:
            # we have remaining examples to retriever (in most few-shot settings, all of them)
            predicted_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)
            modified_item = copy.deepcopy(data_item)
            modified_item['last_slot_values'] = predicted_context
            examples = self.retriever.item_to_best_examples(
                modified_item, k=self.num_examples, decoder=self.demonstration_decoder)
            if isinstance(examples, dict):
                tmp = []
                iter_num = max([len(v) for v in examples.values()])
                # iterate keys in d only for 10 times
                key_iter = itertools.islice(itertools.cycle(examples.keys()), iter_num*len(examples))
                while True:
                    try:
                        key = next(key_iter)
                        tmp.append(examples[key].pop())
                    except IndexError:
                        continue
                    except StopIteration:
                        break
                examples = [e for e in tmp if e['ID'] != data_item['ID']][::-1]
        # verify we haven't selected an example in this dialogue
        examples = [e for e in examples if e['ID'] != data_item['ID']]
        if len(examples) > num_examples:
            examples = examples[-num_examples:]
        return examples

    def run(self) -> Tuple[List[Turn], Dict[str, Any]]:
        jga_by_turn_id = defaultdict(list)  # use to record the accuracy

        selected_set: List[Turn] = self.test_set
        # if needed, only evaluate on particular turns (analysis purpose)
        if self.only_use_turn_idx >= 0:
            if not self.use_gold:
                raise ValueError("can only evaluate particular turn when using gold context")
            selected_set = [d for d in self.test_set if len(d['dialog']['usr']) == self.only_use_turn_idx + 1]

        # start experiment
        running_log: List[Turn] = []
        n_total: int = 0
        n_correct: int = 0
        total_acc: float = 0
        total_f1: float = 0
        train_by_dial_id = group_by_dial_id_and_turn(self.train_set)

        for data_item_idx, data_item in tqdm(enumerate(selected_set)):
            n_total += 1
            
            examples: List[Turn] = self.get_demonstrations(data_item) 
            prompt_text_dict: Final[str] = self.get_prompt_text_dict(data_item, examples, zero_one_shot=False)
            
            # record the prompt
            data_item['prompt'] = prompt_text_dict

            predicted_slot_values: MultiWOZDict = {}
            all_predicted_slot_values: MultiWOZDict = {}
            predicted_prior_context: MultiWOZDict = None
            try:
                dial_id, turn_id = data_item['ID'], data_item['turn_id']
                if self.run_log_cache and dial_id in self.run_log_cache:
                    cache_result: Turn = copy.deepcopy(self.run_log_cache[dial_id][turn_id])
                    assert 'completion' in cache_result
                    logging.info(f"Found a cached turn {dial_id}-{turn_id}, re-using completion: {cache_result['completion']}")
                    best_completion = cache_result['completion']
                    completions = cache_result.get('all_completions')
                    examples = [train_by_dial_id[ex_dial_id][ex_turn_id] for ex_dial_id, ex_turn_id in cache_result['examples']]
                    if 'log_prob_given_null' in cache_result:
                        data_item['log_prob_given_null'] = cache_result['log_prob_given_null']
                    if 'log_probs' in cache_result:
                        data_item['log_probs'] = cache_result['log_probs']
                    if 'completion_to_canonical' in cache_result:
                        data_item['completion_to_canonical'] = cache_result['completion_to_canonical']
                else:
                    # get completion from language model
                    all_completions = {}
                    all_best_completions = {}                   
                    for ids, prompt_text in prompt_text_dict.items():
                        completions = None  # the except block will print it, which can be confusing if its from the previous turn
                        completions, examples = self.generate_completion(prompt_text, data_item, examples)
                        best_completion = max(completions, key=completions.get)
                        best_completion = best_completion.strip().replace('agent.state.', '')
                        all_completions[ids] = completions
                        all_best_completions[ids] = best_completion

                # aggregate the prediction and the history states
                predicted_prior_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)
                predicted_slot_values = self.completion_parser(best_completion, predicted_prior_context)
                ############# NEW CODE #############
                # for ids, best_completion in all_best_completions.items():
                #     all_predicted_slot_values[ids] = self.completion_parser(best_completion, predicted_prior_context)

            except Exception as e:
                print(f"the output could not be parsed successfully: {best_completion}", e)
                data_item['not_valid'] = 1
                data_item['completion'] = best_completion
            predicted_slot_values = self.normalizer.normalize(predicted_slot_values)
            
            ############# NEW CODE #############
            # for ids, predicted_slot_values in all_predicted_slot_values.items():
            #     predicted_slot_values = self.normalizer.normalize(predicted_slot_values)
            #     all_predicted_slot_values[ids] = predicted_slot_values
            ############# NEW CODE #############

            # merge context and prediction
            if self.use_gold:
                prior_dialogue_state = data_item['last_slot_values'].copy()
            else:
                prior_dialogue_state = self.prediction_recorder.retrieve_previous_turn_state(data_item).copy()

            ############# NEW CODE #############
            # _, gold_state_python = python_demo.get_python_statements(data_item['last_slot_values'], data_item['slot_values'], 
            #                                                 turn_strings=[
            #                                                      data_item['dialog']['sys'][-1],
            #                                                      python_demo.get_user_string(data_item['dialog']['usr'][-1])],
            #                                                 detailed_state_string=True)
            # gold_state_python = gold_state_python.replace('agent.state.', '')
            ############# NEW CODE #############

            all_slot_values = update_dialogue_state(prior_dialogue_state, predicted_slot_values)

            # some slots may contain multiple values
            all_slot_values = {k: v.split('|')[0] for k, v in all_slot_values.items()}

            # record current turn prediction
            self.prediction_recorder.add_state(data_item, all_slot_values)

            # ############# NEW CODE #############
            # data_item['gold_completion'] = gold_state_python
            # data_item['all_shots_completions'] = all_completions
            # data_item['all_shots_best_completion'] = all_best_completions
            # data_item['all_predicted_slot_values'] = copy.deepcopy(all_predicted_slot_values)
            # # evaluate the result
            # if len(all_predicted_slot_values) > 1:
            #     zero_shot_pred = all_predicted_slot_values.pop('0-shot', None)
            #     few_shot_pred = all_predicted_slot_values.pop(f"{len(examples)}-shot", None)
            #     assert len(all_predicted_slot_values) == len(examples)

            # import tiktoken
            # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
            # zero_scores = {
            #     'f1': compute_prf(data_item['slot_values'], zero_shot_pred)[0], 
            #     'acc': compute_acc(data_item['slot_values'], zero_shot_pred),
            #     'python-token-f1': calculate_token_f1(encoding, gold_state_python, all_best_completions['0-shot'])
            # }
            
            # score_dict = {ids:{
            #     'f1': compute_prf(data_item['slot_values'], pred_state)[0],
            #     'acc': compute_acc(data_item['slot_values'], pred_state),
            #     'python-token-f1': calculate_token_f1(encoding, gold_state_python, all_best_completions[ids]),
            #     'idx': idx}
            #     for idx, (ids, pred_state) in enumerate(all_predicted_slot_values.items())}
            # assert len(score_dict) == len(examples) 

            # for score_name, zero_score in zero_scores.items():
            #     sorted_score_list = sorted(score_dict.items(), key=lambda x: x[1][score_name], reverse=True)
            #     data_item[score_name] = defaultdict(list)
            #     for ids, score_idx in sorted_score_list:
            #         if score_idx[score_name] > zero_score:
            #             data_item[score_name]['pos'].append(examples[score_idx['idx']])
            #         else:
            #             data_item[score_name]['neg'].append(examples[score_idx['idx']])                
            
            # with open(os.path.join(self.output_dir, "pos_neg.json"), 'a') as f:
            #     if data_item_idx == 0:
            #         f.write('[\n')
            #     json.dump(data_item, f, indent=4)
            #     if data_item_idx < len(selected_set) - 1:
            #         f.write(',\n')
            #     elif data_item_idx == len(selected_set) - 1:
            #         f.write('\n]')
            # ############# NEW CODE #############
            
            # record the predictions
            data_item['pred'] = all_slot_values
            data_item['pred_delta_slot_values'] = predicted_slot_values
            data_item['pred_prior_context'] = predicted_prior_context or {}
            data_item['completion'] = best_completion
            data_item['all_completions'] = completions
            data_item['num_solutions'] = len(completions)
            data_item['prompt_counts'] = count_prompts_from_examples(examples)
            data_item['examples'] = [(e['ID'], e['turn_id']) for e in examples]
            running_log.append(data_item)

            # print the result
            # ############# NEW CODE #############
            # print()
            # all_predicted_slot_values['0-shot'] = zero_shot_pred
            # all_predicted_slot_values[f"{len(examples)}-shot"] = few_shot_pred
            # for ids, completions in all_completions.items():
            #     print(f"{ids} completions: {completions}")
            #     print(f"{ids} best completion: {all_best_completions[ids]}")
            #     print(f"{ids} pred turn change: {pprint.pformat(all_predicted_slot_values[ids])}")
            # ############# NEW CODE #############
            
            print(f"\nfew-shot completions: {completions}")
            print(f"few-shot best completion: {best_completion}")
            print(f"this is the {n_total - 1}th example. {data_item['ID']}_turn_{data_item['turn_id']}")
            print(f"system response: {data_item['dialog']['sys'][-1]}")
            print(f"user response: {data_item['dialog']['usr'][-1]}")
            print(f"pred turn change: {pprint.pformat(predicted_slot_values)}")
            print(f"gold turn change: {pprint.pformat(data_item['turn_slot_values'])}")
            print(f"pred states: {pprint.pformat(all_slot_values)}")
            print(f"gold states: {pprint.pformat(data_item['slot_values'])}")

            this_jga, this_acc, this_f1 = evaluate(all_slot_values, data_item['slot_values'])
            total_acc += this_acc
            total_f1 += this_f1

            if this_jga:
                n_correct += 1
                jga_by_turn_id[data_item['turn_id']].append(1)
                print("\n=====================correct!=======================")
            else:
                jga_by_turn_id[data_item['turn_id']].append(0)
                print("\n=====================wrong!=======================")
            self.logger.log({"current_jga": n_correct / n_total, "n_total": n_total})
            self.logger.step()
            print("\n")

            # write out running log regularly, in-case we stop a run. Give some buffer in-case we accidentally start,
            # and didn't want to over-write
            if data_item_idx > 20:
                with open(os.path.join(self.output_dir, "running_log.json"), 'w') as f:
                    json.dump(running_log, f)
        stats = evaluate_run_log.evaluate_logs(running_log, test_set=self.test_set)
        slot_prf = slot_level_f1(running_log, tp_means_correct=True)
        self.logger.log({f"f1/{slot_name}": f1 for slot_name, (_, f1) in slot_prf.items()})
        self.logger.log({f"precision/{slot_name}": calc_prf(f1_dict).precision for slot_name, (f1_dict, f1) in slot_prf.items()})
        self.logger.log({f"recall/{slot_name}": calc_prf(f1_dict).recall for slot_name, (f1_dict, f1) in slot_prf.items()})

        turn_acc_table: wandb.Table = wandb.Table(data=[
            [f"{turn_id}", acc] for turn_id, acc in stats['turn_accuracies'].items()
        ], columns=['turn_id', 'accuracy'])
        stats['turn_accuracies'] = wandb.plot.bar(turn_acc_table, "turn_id", "accuracy", title="accuracy by turn id")
        self.logger.log(stats)

        # get per-domain stats
        by_domain_stats: Dict[str, Dict[str, Any]] = evaluate_run_log.evaluate_on_domains(running_log, self.test_set)
        flattened_domain_stats: Dict[str, Any] = {}
        for domain, domain_scores in by_domain_stats.items():
            for metric, value in domain_scores.items():
                flattened_domain_stats[f"{domain}-{metric}"] = value
        self.logger.log(flattened_domain_stats)

        # log running_log as an artifact
        self.logger.step()
        return running_log, stats

    @abc.abstractmethod
    def generate_completion(self, prompt_text: str, data_item: Turn, examples: List[Turn]) -> Tuple[
        Dict[str, float], List[Turn]]:
        """
        Given a prompt, a set of demonstration examples, and an example turn, generate the relavant completion for a
        language model (first returned item) and the sub-set of example turns used to generate it (e.g. if the context
        length is exceeded when using all examples given as an argument)

        :param prompt_text: text of the prompt
        :param data_item: current turn
        :param examples: examples to use, 'nearest' in relevancy to the current turn is last
        :return: a dictionary of string completions and their scores. Maximum score should give best completion
                (e.g. log-probability). How many are returned depends on experiment context (greedy, top-p, MI, etc.)
        """
        pass


def get_retriever_by_type(retriever_type: str, retriever_dir: str, retriever_args: Dict[str, Any]) -> ExampleRetriever:
    if retriever_type == "EmbeddingRetriever":
        retriever_full_path: str = get_output_dir_full_path(retriever_dir)
        if retriever_full_path != retriever_dir:
            if not os.path.exists(retriever_full_path):
                raise ValueError(f"retriever_dir={retriever_dir} is a relative path, attempted to find rooted from "
                                 f"{get_output_dir_full_path('')}, but not found. Set an abs path to retriever or "
                                 f"check setting of REFPYDST_OUTPUTS_DIR.")
            logging.info(f"loading retriever from {retriever_full_path}")
        return EmbeddingRetriever(**{
            "model_path": retriever_full_path,
            "search_index_filename": os.path.join(retriever_full_path, "train_index.npy"),
            **retriever_args
        })
    elif retriever_type == "no_retriever":
        return MockRetriever()
    elif retriever_type == "random":
        return RandomExampleRetriever(**retriever_args)
    elif retriever_type == "BM25":
        return BM25Retriever(**retriever_args)
    elif retriever_type == "Mixed":
        retriever_full_path: str = get_output_dir_full_path(retriever_dir)
        return MixedRetriever(**{
            "model_path": retriever_full_path,
            "search_index_filename": os.path.join(retriever_full_path, "train_index.npy"),
            **retriever_args
        })
    elif retriever_type == "Error":
        retriever_full_path: str = get_output_dir_full_path(retriever_dir)
        return ErrorRetriever(**{
            "model_path": retriever_full_path,
            "search_index_filename": os.path.join(retriever_full_path, "train_index.npy"),
            **retriever_args
        })
    else:
        raise ValueError(f"unknown retriever type: {retriever_type}")


def _turn_list_to_dial_id_turn_id_dict(turns: List[Turn]) -> Dict[str, Dict[int, Turn]]:
    turn_dict: Dict[str, Dict[int, Turn]] = defaultdict(dict)
    for turn in turns:
        turn_dict[turn['ID']][turn['turn_id']] = turn
    return turn_dict
