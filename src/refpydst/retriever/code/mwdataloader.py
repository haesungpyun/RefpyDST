from typing import Callable, Union, List

import numpy as np

import argparse
import os
import sys

from refpydst.data_types import Turn, MultiWOZDict, RetrieverFinetuneRunConfig

import wandb

from sentence_transformers import SentenceTransformer, models, InputExample
from sentence_transformers.losses import *
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from refpydst.retriever.code.embed_based_retriever import EmbeddingRetriever
from refpydst.retriever.code.index_based_retriever import IndexRetriever
from refpydst.retriever.code.pretrained_embed_index import embed_everything
from refpydst.retriever.code.retriever_evaluation import evaluate_retriever_on_dataset
from refpydst.retriever.code.st_evaluator import RetrievalEvaluator
from refpydst.utils.general import read_json, get_output_dir_full_path, REFPYDST_OUTPUTS_DIR, read_json_from_data_dir, \
    WANDB_ENTITY, WANDB_PROJECT

from refpydst.prompt_formats.python.demo import get_state_reference
from refpydst.prompt_formats.python.demo import normalize_to_domains_and_slots, SLOT_NAME_REVERSE_REPLACEMENTS
from refpydst.retriever.code.retriever_evaluation import compute_sv_sim
from refpydst.utils.general import read_json_from_data_dir

# Only care domain in test

DOMAINS = ['hotel', 'restaurant', 'attraction', 'taxi', 'train']


def important_value_to_string(slot, value):
    if value in ["none", "dontcare"]:
        return f"{slot}{value}"  # special slot
    return f"{slot}-{value}"


StateTransformationFunction = Callable[[Turn], Union[List[str], MultiWOZDict]]


def default_state_transform(turn: Turn) -> List[str]:
    return [important_value_to_string(s, v) for s, v in turn['turn_slot_values'].items()
            if s.split('-')[0] in DOMAINS]


def reference_aware_state_transform(turn: Turn) -> List[str]:
    context_slot_values: MultiWOZDict = {s: v.split('|')[0] for s, v in turn['last_slot_values'].items()}
    context_domains_to_slot_pairs = normalize_to_domains_and_slots(context_slot_values)
    turn_domains_to_slot_pairs = normalize_to_domains_and_slots(turn['turn_slot_values'])
    user_str = turn['dialog']['usr'][-1]
    sys_str = turn['dialog']['sys'][-1]
    sys_str = sys_str if sys_str and not sys_str == 'none' else ''
    new_dict: MultiWOZDict = {}
    for domain, slot_pairs in turn_domains_to_slot_pairs.items():
        for norm_slot_name, norm_slot_value in slot_pairs.items():
            # Because these come from a prompting template set up, the slot names were changed in a few cases,
            # and integer-string values were turned into actual integers
            mwoz_slot_name = SLOT_NAME_REVERSE_REPLACEMENTS.get(norm_slot_name, norm_slot_name.replace("_", " "))
            mwoz_slot_value = str(norm_slot_value)
            reference = get_state_reference(context_domains_to_slot_pairs, domain, norm_slot_name, norm_slot_value,
                                            turn_strings=[sys_str, user_str])
            if reference is not None:
                referred_domain, referred_slot = reference
                new_dict[f"{domain}-{mwoz_slot_name}"] = f"state.{referred_domain}.{referred_slot}"
            else:
                new_dict[f"{domain}-{mwoz_slot_name}"] = mwoz_slot_value
    return [important_value_to_string(s, v) for s, v in new_dict.items()
            if s.split('-')[0] in DOMAINS]


def get_state_transformation_by_type(tranformation_type: str) -> StateTransformationFunction:
    if not tranformation_type or tranformation_type == "default":
        return default_state_transform
    elif tranformation_type == "ref_aware":
        return reference_aware_state_transform
    else:
        raise ValueError(f"Unsupported transformation type: {tranformation_type}")


def get_string_transformation_by_type(tranformation_type: str):
    if not tranformation_type or tranformation_type == "default":
        return data_item_to_string
    else:
        raise ValueError(f"Unsupported transformation type: {tranformation_type}")


def input_to_string(context_dict, sys_utt, usr_utt):
    history = state_to_NL(context_dict)
    if sys_utt == 'none':
        sys_utt = ''
    if usr_utt == 'none':
        usr_utt = ''
    history += f" [SYS] {sys_utt} [USER] {usr_utt}"
    return history


def data_item_to_string(
    data_item: Turn, 
    string_transformation=input_to_string, 
    full_history: bool = False, 
    **kwargs
) -> str:
    """
    Converts a turn to a string with the context, system utterance, and user utterance in order

    :param data_item: turn to use
    :param string_transformation: function defining how to represent the context, system utterance, user utterance
           triplet as a single string
    :param full_history: use the complete dialogue history, not just current turn
    :return: string representation, like below


    Example (new lines and tabs added for readability):
    [CONTEXT] attraction name: saint johns college,
    [SYS] saint john s college is in the centre of town on saint john s street . the entrance fee is 2.50 pounds .
          can i help you with anything else ?
    [USER] is there an exact address , like a street number ? thanks !
    """
    input_type = kwargs.get('input_type', 'dialog_context')
    only_slot = kwargs.get('only_slot', False)
    
    if input_type == 'dialog_context':
        # use full history, depend on retriever training (for ablation)
        # orginal code
        if full_history:
            history = ""
            for sys_utt, usr_utt in zip(data_item['dialog']['sys'], data_item['dialog']['usr']):
                history += string_transformation({}, sys_utt, usr_utt)
            return history

        # if full_history:
        #     history = "[CONTEXT] "
        #     for sys_utt, usr_utt in zip(data_item['dialog']['sys'], data_item['dialog']['usr']):
        #         history += f" [SYS] {sys_utt} [USER] {usr_utt}"
        #     return history
        
        # Use only slots in the context
        if only_slot:
            context = list(data_item['last_slot_values'].keys())
            sys_utt = data_item['dialog']['sys'][-1]
            usr_utt = data_item['dialog']['usr'][-1]
            history = "[CONTEXT] " + ', '.join(context)
            if sys_utt == 'none':
                sys_utt = ''
            if usr_utt == 'none':
                usr_utt = ''
            history += f" [SYS] {sys_utt} [USER] {usr_utt}"
            return history

        # use single turn
        context = data_item['last_slot_values']
        sys_utt = data_item['dialog']['sys'][-1]
        usr_utt = data_item['dialog']['usr'][-1]
        history = string_transformation(context, sys_utt, usr_utt)
        return history
    
    elif input_type == 'context':
        if full_history:
            history = ""
            for sys_utt, usr_utt in zip(data_item['dialog']['sys'][:-1], data_item['dialog']['usr'][:-1]):
                history += string_transformation({}, sys_utt, usr_utt)
            return history
         
        if only_slot:
            context = list(data_item['last_slot_values'].keys())
            history = "[CONTEXT] " + ', '.join(context)
            return history
        
        context = data_item['last_slot_values']
        history = string_transformation(context, '', '')
        return history
    
    elif input_type == 'dialog':
        sys_utt = data_item['dialog']['sys'][-1]
        usr_utt = data_item['dialog']['usr'][-1]
        history = string_transformation({}, sys_utt, usr_utt)
        return history
    
    return history


def state_to_NL(slot_value_dict):
    output = "[CONTEXT] "
    for k, v in slot_value_dict.items():
        output += f"{' '.join(k.split('-'))}: {v.split('|')[0]}, "
    return output

class MWDataset2:

    def __init__(self, mw_json_fn: str, just_embed_all=False, beta: float = 1.0,
                 string_transformation: Callable[[Turn], str] = data_item_to_string,
                 state_transformation: StateTransformationFunction = default_state_transform):

        data = read_json_from_data_dir(mw_json_fn)

        self.turn_labels = []  # store [SMUL1843.json_turn_1, ]
        self.turn_utts = []  # store corresponding text
        self.turn_states = []  # store corresponding states. [['attraction-type-mueseum',],]
        self.string_transformation = string_transformation
        self.state_transformation = state_transformation

        for turn in data:
            # filter the domains that not belongs to the test domain
            if not set(turn["domains"]).issubset(set(DOMAINS)):
                continue

            # update dialogue history
            history = self.string_transformation(turn)

            # convert to list of strings
            current_state = self.state_transformation(turn)

            self.turn_labels.append(f"{turn['ID']}_turn_{turn['turn_id']}")
            self.turn_utts.append(history)
            self.turn_states.append(current_state)

        self.n_turns = len(self.turn_labels)
        print(f"there are {self.n_turns} turns in this dataset")

        if not just_embed_all:
            # compute all similarity
            self.similarity_matrix = np.zeros((self.n_turns, self.n_turns))
            for i in tqdm(range(self.n_turns)):
                self.similarity_matrix[i, i] = 1
                for j in range(i, self.n_turns):
                    self.similarity_matrix[i, j] = compute_sv_sim(self.turn_states[i],
                                                                  self.turn_states[j],
                                                                  beta=beta)
                    self.similarity_matrix[j, i] = self.similarity_matrix[i, j]

class MWDataset:

    def __init__(self, mw_json_fn: str, just_embed_all=False, beta: float = 1.0,
                 string_transformation: Callable[[Turn], str] = data_item_to_string,
                 state_transformation: StateTransformationFunction = default_state_transform):

        data = read_json_from_data_dir(mw_json_fn)

        self.turn_labels = []  # store [SMUL1843.json_turn_1, ]
        self.turn_utts = []  # store corresponding text
        self.turn_states = []  # store corresponding states. [['attraction-type-mueseum',],]
        self.string_transformation = string_transformation
        self.state_transformation = state_transformation
        self.turn_scores = []
        self.best_example = []

        for turn in data:
            # filter the domains that not belongs to the test domain
            if not set(turn["domains"]).issubset(set(DOMAINS)):
                continue

            # update dialogue history
            history = self.string_transformation(turn)

            # convert to list of strings
            current_state = self.state_transformation(turn)

            self.turn_labels.append(f"{turn['ID']}_turn_{turn['turn_id']}")
            self.turn_utts.append(history)
            self.turn_states.append(current_state)
            self.turn_scores.append(turn['final_scores'])
            best_examples=[]
            for example in turn['best_example']:
                best_example_label = f"{example['ID']}_turn_{example['turn_id']}"
                best_examples.append(best_example_label)
            self.best_example.append(best_examples)


        self.n_turns = len(self.turn_labels)
        print(f"there are {self.n_turns} turns in this dataset")


def save_embeddings(model, dataset: MWDataset, output_filename: str) -> None:
    """
    Save embeddings for all items in the dataset to the output_filename

    :param dataset: dataset to create and save embeddings from
    :param output_filename: path to the save file
    :return: None
    """
    embeddings = model.encode(dataset.turn_utts, convert_to_numpy=True)
    output = {}
    for i in tqdm(range(len(embeddings))):
        output[dataset.turn_labels[i]] = embeddings[i:i + 1]
    np.save(output_filename, output)

def save_embeddings2(model, dataset: MWDataset2, output_filename: str) -> None:
    """
    Save embeddings for all items in the dataset to the output_filename

    :param dataset: dataset to create and save embeddings from
    :param output_filename: path to the save file
    :return: None
    """
    embeddings = model.encode(dataset.turn_utts, convert_to_numpy=True)
    output = {}
    for i in tqdm(range(len(embeddings))):
        output[dataset.turn_labels[i]] = embeddings[i:i + 1]
    np.save(output_filename, output)

class MWContrastiveDataloader:
    """
    Constrastive Learning Data Loader w/ hard-negative sampling, from:

    @article{hu2022context,
      title={In-Context Learning for Few-Shot Dialogue State Tracking},
      author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
      journal={arXiv preprint arXiv:2203.08568},
      year={2022}
    }
    """

    def __init__(self, f1_set: MWDataset2, pretrained_retriever: IndexRetriever):
        """

        :param f1_set:
        :param pretrained_retriever:
        """
        self.f1_set = f1_set
        self.pretrained_retriever = pretrained_retriever

    def hard_negative_sampling(self, topk=10, top_range=100):
        sentences1 = []
        sentences2 = []
        scores = []

        # do hard negative sampling
        for ind in tqdm(range(self.f1_set.n_turns)):

            # find nearest neighbors given by pre-trained retriever
            this_label = self.f1_set.turn_labels[ind]
            nearest_labels = self.pretrained_retriever.label_to_nearest_labels(
                this_label, k=top_range + 1)[:-1]  # to exclude itself
            nearest_args = [self.f1_set.turn_labels.index(
                l) for l in nearest_labels]

            # topk and bottomk nearest f1 score examples, as hard examples
            similarities = self.f1_set.similarity_matrix[ind][nearest_args]
            sorted_args = similarities.argsort()

            chosen_positive_args = list(sorted_args[-topk:])
            chosen_negative_args = list(sorted_args[:topk])

            chosen_positive_args = np.array(nearest_args)[chosen_positive_args]
            chosen_negative_args = np.array(nearest_args)[chosen_negative_args]

            for chosen_arg in chosen_positive_args:
                sentences1.append(self.f1_set.turn_utts[ind])
                sentences2.append(self.f1_set.turn_utts[chosen_arg])
                scores.append(1)

            for chosen_arg in chosen_negative_args:
                sentences1.append(self.f1_set.turn_utts[ind])
                sentences2.append(self.f1_set.turn_utts[chosen_arg])
                scores.append(0)

        return sentences1, sentences2, scores

    def generate_eval_examples(self, topk=5, top_range=100):
        # add topk closest, furthest, and n_random random indices
        sentences1, sentences2, scores = self.hard_negative_sampling(
            topk=topk, top_range=top_range)
        scores = [float(s) for s in scores]
        return sentences1, sentences2, scores

    def generate_train_examples(self, topk=5, top_range=100):
        sentences1, sentences2, scores = self.generate_eval_examples(
            topk=topk, top_range=top_range)
        n_samples = len(sentences1)
        return [InputExample(texts=[sentences1[i], sentences2[i]], label=scores[i])
                for i in range(n_samples)]
                


class CLDataloader:
    """
    pos: top 10
    neg: bot 10
    """
    def __init__(self, f1_set: MWDataset, score_type: str):
        self.f1_set = f1_set
        self.label_to_index = {label: idx for idx, label in enumerate(self.f1_set.turn_labels)}  
        self.score_type = score_type

    def hard_negative_sampling(self, topk=10):
        sentences1 = []
        sentences2 = []
        scores = []

        # do hard negative sampling
        for ind in tqdm(range(self.f1_set.n_turns)):
            if self.score_type == "simple":
                score_full = self.f1_set.turn_scores[ind]['score_full']
            elif self.score_type == "influence":
                score_full = self.f1_set.turn_scores[ind]['influence_full']
            elif self.score_type == "simple_delta":
                score_full = self.f1_set.turn_scores[ind]['score_delta']
            elif self.score_type == "influence_delta":
                score_full = self.f1_set.turn_scores[ind]['influence_delta']
            else:
                raise ValueError(f"Unknown score type: {self.score_type}")
            sorted_scores = sorted(score_full.items(), key=lambda item: item[1], reverse=True)

            chosen_positive_args = sorted_scores[:topk]
            chosen_negative_args = sorted_scores[-topk:]

            for label, score in chosen_positive_args:
                idx = self.label_to_index[label]  
                sentences1.append(self.f1_set.turn_utts[ind])
                sentences2.append(self.f1_set.turn_utts[idx])
                scores.append(1)

            for label, score in chosen_negative_args:
                idx = self.label_to_index[label]  
                sentences1.append(self.f1_set.turn_utts[ind])
                sentences2.append(self.f1_set.turn_utts[idx])
                scores.append(0)

        positive_count = scores.count(1)
        negative_count = scores.count(0)

        print(f"Number of positive samples: {positive_count}")
        print(f"Number of negative samples: {negative_count}")

        return sentences1, sentences2, scores

    def generate_eval_examples(self, topk=5):
        # add topk closest, furthest, and n_random random indices
        sentences1, sentences2, scores = self.hard_negative_sampling(
            topk=topk)
        scores = [float(s) for s in scores]
        return sentences1, sentences2, scores

    def generate_train_examples(self, topk=5):
        sentences1, sentences2, scores = self.generate_eval_examples(
            topk=topk)
        n_samples = len(sentences1)
        return [InputExample(texts=[sentences1[i], sentences2[i]], label=scores[i])
                for i in range(n_samples)]


class MWpos3neg012Dataloader:
    """
    pos: sampling 10 randomly in score_full=3
    neg: sampling 10 randomly in score_full=0,1,2
    """
    def __init__(self, f1_set: MWDataset):
        self.f1_set = f1_set
        self.label_to_index = {label: idx for idx, label in enumerate(self.f1_set.turn_labels)}  

    def hard_negative_sampling(self, topk=10):
        sentences1 = []
        sentences2 = []
        scores = []

        # do hard negative sampling
        for ind in tqdm(range(self.f1_set.n_turns)):
            score_full = self.f1_set.turn_scores[ind]['score_full']
            
            # Positive: score_full == 3
            positive_args = [(label, score) for label, score in score_full.items() if score == 3]
            # Negative: score_full in [0, 1, 2]
            negative_args = [(label, score) for label, score in score_full.items() if score in [0, 1, 2]]

            # Sample topk positives and topk negatives
            chosen_positive_args = positive_args[:topk]
            chosen_negative_args = negative_args[:topk]

            for label, score in chosen_positive_args:
                idx = self.label_to_index[label]  
                sentences1.append(self.f1_set.turn_utts[ind])
                sentences2.append(self.f1_set.turn_utts[idx])
                scores.append(1)

            for label, score in chosen_negative_args:
                idx = self.label_to_index[label]  
                sentences1.append(self.f1_set.turn_utts[ind])
                sentences2.append(self.f1_set.turn_utts[idx])
                scores.append(0)

        positive_count = scores.count(1)
        negative_count = scores.count(0)

        print(f"Number of positive samples: {positive_count}")
        print(f"Number of negative samples: {negative_count}")

        return sentences1, sentences2, scores

    def generate_eval_examples(self, topk=5):
        # add topk closest, furthest, and n_random random indices
        sentences1, sentences2, scores = self.hard_negative_sampling(
            topk=topk)
        scores = [float(s) for s in scores]
        return sentences1, sentences2, scores

    def generate_train_examples(self, topk=5):
        sentences1, sentences2, scores = self.generate_eval_examples(
            topk=topk)
        n_samples = len(sentences1)
        return [InputExample(texts=[sentences1[i], sentences2[i]], label=scores[i])
                for i in range(n_samples)]

class MWpos3neg0Dataloader:
    """
    pos: sampling 10 randomly in score_full=3
    neg: sampling 10 randomly in score_full=0
    """
    def __init__(self, f1_set: MWDataset):
        self.f1_set = f1_set
        self.label_to_index = {label: idx for idx, label in enumerate(self.f1_set.turn_labels)}  

    def hard_negative_sampling(self, topk=10):
        sentences1 = []
        sentences2 = []
        scores = []

        # do hard negative sampling
        for ind in tqdm(range(self.f1_set.n_turns)):
            score_full = self.f1_set.turn_scores[ind]['score_full']
            
            # Positive: score_full == 3
            positive_args = [(label, score) for label, score in score_full.items() if score == 3]
            # Negative: score_full in [0, 1, 2]
            negative_args = [(label, score) for label, score in score_full.items() if score == 0]

            # Sample topk positives and topk negatives
            chosen_positive_args = positive_args[:topk]
            chosen_negative_args = negative_args[:topk]

            for label, score in chosen_positive_args:
                idx = self.label_to_index[label]  
                sentences1.append(self.f1_set.turn_utts[ind])
                sentences2.append(self.f1_set.turn_utts[idx])
                scores.append(1)

            for label, score in chosen_negative_args:
                idx = self.label_to_index[label]  
                sentences1.append(self.f1_set.turn_utts[ind])
                sentences2.append(self.f1_set.turn_utts[idx])
                scores.append(0)

        positive_count = scores.count(1)
        negative_count = scores.count(0)

        print(f"Number of positive samples: {positive_count}")
        print(f"Number of negative samples: {negative_count}")

        return sentences1, sentences2, scores

    def generate_eval_examples(self, topk=5):
        # add topk closest, furthest, and n_random random indices
        sentences1, sentences2, scores = self.hard_negative_sampling(
            topk=topk)
        scores = [float(s) for s in scores]
        return sentences1, sentences2, scores

    def generate_train_examples(self, topk=5):
        sentences1, sentences2, scores = self.generate_eval_examples(
            topk=topk)
        n_samples = len(sentences1)
        return [InputExample(texts=[sentences1[i], sentences2[i]], label=scores[i])
                for i in range(n_samples)]

class MWtripleDataloader:
    """
    pos: top 5 in sorted score_full
    neg: bot 5 in sorted score_full
    """
    def __init__(self, f1_set: MWDataset, score_type: str):
        self.f1_set = f1_set
        self.label_to_index = {label: idx for idx, label in enumerate(self.f1_set.turn_labels)}  
        self.score_type = score_type

    def hard_negative_sampling(self, topk=10):
        anchor = []
        positive = []
        negative = []

        # do hard negative sampling
        for ind in tqdm(range(self.f1_set.n_turns)):
            if self.score_type == "simple":
                score_full = self.f1_set.turn_scores[ind]['score_full']
            elif self.score_type == "influence":
                score_full = self.f1_set.turn_scores[ind]['influence_full']
            elif self.score_type == "simple_delta":
                score_full = self.f1_set.turn_scores[ind]['score_delta']
            elif self.score_type == "influence_delta":
                score_full = self.f1_set.turn_scores[ind]['influence_delta']
            else:
                raise ValueError(f"Unknown score type: {self.score_type}")
            sorted_scores = sorted(score_full.items(), key=lambda item: item[1], reverse=True)

            chosen_positive_args = sorted_scores[:topk]
            chosen_negative_args = sorted_scores[-topk:]

            for label, score in chosen_positive_args:
                idx = self.label_to_index[label]  
                anchor.append(self.f1_set.turn_utts[ind])
                positive.append(self.f1_set.turn_utts[idx])

            for label, score in chosen_negative_args:
                idx = self.label_to_index[label]  
                negative.append(self.f1_set.turn_utts[idx])

        return anchor, positive, negative

    def generate_eval_examples(self, topk=5):
        # add topk closest, furthest, and n_random random indices
        anchor, positive, negative = self.hard_negative_sampling(
            topk=topk)
        return anchor, positive, negative

    def generate_train_examples(self, topk=5):
        anchor, positive, negative = self.generate_eval_examples(
            topk=topk)
        n_samples = len(anchor)
        return [InputExample(texts=[anchor[i], positive[i], negative[i]])
                for i in range(n_samples)]

class MWtriple2Dataloader:
    """
    pos: top 5 in sorted score_full
    neg: bot 5 in sorted score_full
    """
    def __init__(self, f1_set: MWDataset):
        self.f1_set = f1_set
        self.label_to_index = {label: idx for idx, label in enumerate(self.f1_set.turn_labels)}  

    def hard_negative_sampling(self, topk=10):
        anchor = []
        positive = []
        negative = []

        # do hard negative sampling
        for ind in tqdm(range(self.f1_set.n_turns)):
            score_full = self.f1_set.turn_scores[ind]['score_full']
            sorted_scores = sorted(score_full.items(), key=lambda item: item[1], reverse=True)

            chosen_positive_args = sorted_scores[:topk]
            chosen_negative_args = sorted_scores[-topk:]

            # Check if all positive and negative examples have the same score
            if chosen_positive_args and chosen_negative_args:
                # Check if the highest positive score is equal to the lowest negative score
                if chosen_positive_args[-1][1] == chosen_negative_args[0][1]:
                    continue  # Skip this turn if all examples have the same score

            for label, score in chosen_positive_args:
                idx = self.label_to_index[label]  
                anchor.append(self.f1_set.turn_utts[ind])
                positive.append(self.f1_set.turn_utts[idx])

            for label, score in chosen_negative_args:
                idx = self.label_to_index[label]  
                negative.append(self.f1_set.turn_utts[idx])

        return anchor, positive, negative

    def generate_eval_examples(self, topk=5):
        # add topk closest, furthest, and n_random random indices
        anchor, positive, negative = self.hard_negative_sampling(
            topk=topk)
        return anchor, positive, negative

    def generate_train_examples(self, topk=5):
        anchor, positive, negative = self.generate_eval_examples(
            topk=topk)
        n_samples = len(anchor)
        return [InputExample(texts=[anchor[i], positive[i], negative[i]])
                for i in range(n_samples)]

class MWmnrDataloader:
    """
    pos: top 5 in sorted score_full
    neg: bot 5 in sorted score_full
    """
    def __init__(self, f1_set: MWDataset, score_type: str):
        self.f1_set = f1_set
        self.label_to_index = {label: idx for idx, label in enumerate(self.f1_set.turn_labels)}  
        self.score_type = score_type
    def hard_negative_sampling(self, topk=10):
        anchor = []
        positive = []

        # do hard negative sampling
        for ind in tqdm(range(self.f1_set.n_turns)):
            if self.score_type == "simple":
                score_full = self.f1_set.turn_scores[ind]['score_full']
            elif self.score_type == "influence":
                score_full = self.f1_set.turn_scores[ind]['influence_full']
            elif self.score_type == "simple_delta":
                score_full = self.f1_set.turn_scores[ind]['score_delta']
            elif self.score_type == "influence_delta":
                score_full = self.f1_set.turn_scores[ind]['influence_delta']
            else:
                raise ValueError(f"Unknown score type: {self.score_type}")
            sorted_scores = sorted(score_full.items(), key=lambda item: item[1], reverse=True)

            chosen_positive_args = sorted_scores[:topk]

            for label, score in chosen_positive_args:
                idx = self.label_to_index[label]  
                anchor.append(self.f1_set.turn_utts[ind])
                positive.append(self.f1_set.turn_utts[idx])

        return anchor, positive

    def generate_eval_examples(self, topk=5):
        # add topk closest, furthest, and n_random random indices
        anchor, positive = self.hard_negative_sampling(
            topk=topk)
        return anchor, positive

    def generate_train_examples(self, topk=5):
        anchor, positive = self.generate_eval_examples(
            topk=topk)
        n_samples = len(anchor)
        return [InputExample(texts=[anchor[i], positive[i]])
                for i in range(n_samples)]


class MWmnrbestDataloader:
    """
    pos: top 5 in sorted score_full
    neg: bot 5 in sorted score_full
    """
    def __init__(self, f1_set: MWDataset, score_type: str):
        self.f1_set = f1_set
        self.label_to_index = {label: idx for idx, label in enumerate(self.f1_set.turn_labels)}  
        self.score_type = score_type
    def hard_negative_sampling(self, topk=10):
        anchor = []
        positive = []

        # do hard negative sampling
        for ind in tqdm(range(self.f1_set.n_turns)):

            chosen_positive_args = self.f1_set.best_example[ind][:topk]

            for label in chosen_positive_args:
                idx = self.label_to_index[label]
                anchor.append(self.f1_set.turn_utts[ind])
                positive.append(self.f1_set.turn_utts[idx])

        return anchor, positive

    def generate_eval_examples(self, topk=5):
        # add topk closest, furthest, and n_random random indices
        anchor, positive = self.hard_negative_sampling(
            topk=topk)
        return anchor, positive

    def generate_train_examples(self, topk=5):
        anchor, positive = self.generate_eval_examples(
            topk=topk)
        n_samples = len(anchor)
        return [InputExample(texts=[anchor[i], positive[i]])
                for i in range(n_samples)]