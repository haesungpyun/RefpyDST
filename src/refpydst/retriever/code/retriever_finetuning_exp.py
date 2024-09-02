import argparse
import os
import sys

from typing import Callable, Union, List
from refpydst.data_types import Turn, MultiWOZDict, RetrieverFinetuneRunConfig

import numpy as np
import wandb

from sentence_transformers import SentenceTransformer, models, InputExample
from sentence_transformers.losses import *
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mwdataloader import *

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


def main(train_fn: str, dev_fn: str, test_fn: str, output_dir: str, mwdataloader_class: str, loss_function: str,
         score_type: str, pretrained_index_root: str = None,
         pretrained_model_full_name: str = 'sentence-transformers/all-mpnet-base-v2', num_epochs: int = 3,
         top_k: int = 5, top_range: int = 200,
         pooling_mode: str = None, f_beta: float = 1.0, log_wandb_freq: int = 100,
         str_transformation_type: str = "default", state_transformation_type: str = "default", 
         max_seq_length: int = 256, batch_size: int = 32,  **kwargs):
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

    if args["mwdataloader_class"] == "MWpos3neg012Dataloader":
        mwdataloader_class = MWpos3neg012Dataloader
    elif args["mwdataloader_class"] == "MWpos3neg0Dataloader":
        mwdataloader_class = MWpos3neg0Dataloader
    elif args["mwdataloader_class"] == "MWmnrDataloader":
        mwdataloader_class = MWmnrDataloader
    elif args["mwdataloader_class"] == "MWtripleDataloader":
        mwdataloader_class = MWtripleDataloader
    elif args["mwdataloader_class"] == "MWtriple2Dataloader":
        mwdataloader_class = MWtriple2Dataloader
    elif args["mwdataloader_class"] == "MWmnrbestDataloader":
        mwdataloader_class = MWmnrbestDataloader
    else:
        raise ValueError(f"Unknown dataloader class: {args['mwdataloader_class']}")

    mw_train_loader = mwdataloader_class(f1_train_set, score_type=score_type)

    # add special tokens and resize
    tokens = ["[USER]", "[SYS]", "[CONTEXT]"]
    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda:0")

    # prepare training dataloaders
    all_train_samples = mw_train_loader.generate_train_examples(topk=top_k)
    print(all_train_samples[20])
    print()
    print(all_train_samples[40])
    train_dataloader = DataLoader(all_train_samples, shuffle=True, batch_size=batch_size)
    print(f"=====number of batches {len(train_dataloader)}=====")

    evaluator: RetrievalEvaluator = RetrievalEvaluator(train_fn=train_fn, dev_fn=dev_fn, index_set=f1_train_set,
                                                       string_transformation=string_transformation)

    # Training. Loss is constructed base on loss type argument
    if args["loss_function"] == "MultipleNegativesRankingLoss":
        loss_function = MultipleNegativesRankingLoss
    elif args["loss_function"] == "OnlineContrastiveLoss":
        loss_function = OnlineContrastiveLoss
    elif args["loss_function"] == "TripletLoss":
        loss_function = TripletLoss
    elif args["loss_function"] == "MultipleNegativesSymmetricRankingLoss":
        loss_function = MultipleNegativesSymmetricRankingLoss
    else:
        raise ValueError(f"Unknown loss function class: {args['loss_function']}")

    train_loss: nn.Module = loss_function(model=model)

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
    parser = argparse.ArgumentParser(description="Run retriever finetuning.")
    parser.add_argument('--run_file', type=str, required=True, 
                        help="Path to the JSON config file.")
    args_cmd = parser.parse_args()

    run_file = args_cmd.run_file
    args: RetrieverFinetuneRunConfig = read_json(run_file)
    if 'output_dir' not in args:
        args['output_dir'] = get_output_dir_full_path(run_file.replace('.json', ''))
    if 'run_name' not in args:
        args['run_name'] = args['output_dir'].replace(os.environ.get(REFPYDST_OUTPUTS_DIR, "outputs"), "").replace(
            '/', '-')

    default_run_name: str = args['output_dir'].replace("../expts/", "").replace('/', '-')
    default_run_group: str = default_run_name.rsplit('-', maxsplit=1)[0]
    #wandb_entity: str = os.environ.get(WANDB_ENTITY, "hacastle12")
    #wandb_project: str = os.environ.get(WANDB_PROJECT, "refpydst")
    wandb.init(project="retriever_finetuning", entity="wpppqwwe", group=args.get("run_group", default_run_group),
               name=args.get("run_name", default_run_name))
    main(**args)