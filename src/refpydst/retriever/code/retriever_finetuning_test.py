import argparse
import os
import sys

from typing import Callable, Union, List
from refpydst.data_types import Turn, MultiWOZDict, RetrieverFinetuneRunConfig

import numpy as np
import wandb

from sentence_transformers import SentenceTransformer, models, InputExample
from sentence_transformers.losses import OnlineContrastiveLoss
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
    only_slot = kwargs.get('only_slot', True)
    
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

class MWContrastiveDataloader:
    """
    pos: top 10 in sorted score_full
    neg: bot 10 in sorted score_full
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

class MWContrastiveDataloader2:
    """
    pos: sampling 10 randomly in score_full=3
    neg: sampling 10 randomly in score_full=1,2,3
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



def main(train_fn: str, dev_fn: str, test_fn: str, output_dir: str, pretrained_index_root: str = None,
         pretrained_model_full_name: str = 'sentence-transformers/all-mpnet-base-v2', num_epochs: int = 15,
         top_k: int = 10, top_range: int = 200,
         pooling_mode: str = None, f_beta: float = 1.0, log_wandb_freq: int = 100,
         str_transformation_type: str = "default", state_transformation_type: str = "default", 
         max_seq_length: int = 256, batch_size: int = 24, checkpoint_save_steps: int = 100, **kwargs):
    wandb.config = dict(locals())

    train_set: List[Turn] = read_json_from_data_dir(train_fn)
    print("=====train set is loaded=====")

    # prepare the retriever model
    word_embedding_model: models.Transformer = models.Transformer(pretrained_model_full_name, max_seq_length=max_seq_length)
    pooling_model: models.Pooling = models.Pooling(
        word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
        pooling_mode=pooling_mode,
    )

    # Choose transformation (how each turn will be represented as a string for retriever training)
    string_transformation: Callable[[Turn], str] = get_string_transformation_by_type(str_transformation_type)

    state_transformation: StateTransformationFunction = get_state_transformation_by_type(state_transformation_type)

    # Preparing dataset
    f1_train_set = MWDataset(train_fn, beta=f_beta, string_transformation=string_transformation,
                             state_transformation=state_transformation)

    mw_train_loader = MWContrastiveDataloader2(f1_train_set)

    # add special tokens and resize
    tokens = ["[USER]", "[SYS]", "[CONTEXT]"]
    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda:0")

    # prepare training dataloaders
    all_train_samples = mw_train_loader.generate_train_examples(topk=top_k)
    train_dataloader = DataLoader(all_train_samples, shuffle=True, batch_size=batch_size)
    print(f"=====number of batches {len(train_dataloader)}=====")

    evaluator: RetrievalEvaluator = RetrievalEvaluator(train_fn=train_fn, dev_fn=dev_fn, index_set=f1_train_set,
                                                       string_transformation=string_transformation)

    # Training. Loss is constructed base on loss type argument
    train_loss: nn.Module = OnlineContrastiveLoss(model=model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=100,
              evaluator=evaluator, evaluation_steps=(len(train_dataloader) // 3),
              output_path=output_dir)

    # load best model
    model = SentenceTransformer(output_dir, device="cuda:0")

    # Note: previously this would embed all train set items, even those not in the training set. However this would risk
    # later use of this retriever and its indices with data it wasn't trained on that should be outside of its selection
    # pool. For now, not permitting this, and only saving the embeddings for the training set. If needed we can add an
    # explicit argument for the dataset to load and embed.
    save_embeddings(model, f1_train_set, os.path.join(output_dir, "train_index.npy"))
    print("=====saving embedding is completed=====")

    test_set: List[Turn] = read_json_from_data_dir(test_fn)
    print("=====test set is loaded=====")

    model.save(output_dir)
    retriever: EmbeddingRetriever = EmbeddingRetriever(
        datasets=[train_set],
        model_path=output_dir,
        search_index_filename=os.path.join(output_dir, "train_index.npy"),
        sampling_method="pre_assigned",
        string_transformation=string_transformation
    )

    # save the retriever as an artifact
    artifact: wandb.Artifact = wandb.Artifact(wandb.run.name, type="model")
    artifact.add_dir(output_dir)
    wandb.log_artifact(artifact)

    print("=====Now evaluating retriever ...=====")
    turn_sv, turn_s, dial_sv, dial_s = evaluate_retriever_on_dataset(test_set, retriever)
    wandb.log({
        "test_top_5_turn_slot_value_f_score": turn_sv,
        "test_top_5_turn_slot_name_f_score": turn_s,
        "test_top_5_hist_slot_value_f_score": dial_sv,
        "test_top_5_hist_slot_name_f_score": dial_s,
    })


if __name__ == '__main__':
    run_file = "/home/haesungpyun/my_refpydst/jun/ret_toy_test.json"
    args: RetrieverFinetuneRunConfig = read_json(run_file)
    if 'output_dir' not in args:
        args['output_dir'] = get_output_dir_full_path(run_file.replace('.json', ''))
    if 'run_name' not in args:
        args['run_name'] = args['output_dir'].replace(os.environ.get(REFPYDST_OUTPUTS_DIR, "outputs"), "").replace(
            '/', '-')

    default_run_name: str = args['output_dir'].replace("../expts/", "").replace('/', '-')
    default_run_group: str = default_run_name.rsplit('-', maxsplit=1)[0]
    wandb_entity: str = os.environ.get(WANDB_ENTITY, "hacastle12")
    wandb_project: str = os.environ.get(WANDB_PROJECT, "refpydst")
    wandb.init(project=wandb_project, entity=wandb_entity, group=args.get("run_group", default_run_group),
               name=args.get("run_name", default_run_name))
    main(**args)
