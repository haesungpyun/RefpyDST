"""
Methods for generating language model completions with the Codex family of models via the OpenAI API.

This file was adapted from the code for the paper "In Context Learning for Dialogue State Tracking", as originally
published here: https://github.com/Yushi-Hu/IC-DST. Cite their article as:

@article{hu2022context,
  title={In-Context Learning for Few-Shot Dialogue State Tracking},
  author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
  journal={arXiv preprint arXiv:2203.08568},
  year={2022}
}
"""
import logging
import os
from typing import List, TypedDict, Optional, Dict, Any, Callable
import torch

from transformers import AutoTokenizer, LlamaForCausalLM

import openai
from openai import BadRequestError
from openai._exceptions import RateLimitError, APIError, APIConnectionError, OpenAIError

from accelerate import Accelerator
from transformers import BitsAndBytesConfig

from refpydst.abstract_lm_client import AbstractLMClient
from refpydst.utils.general import check_argument
from refpydst.utils.speed_limit_timer import SpeedLimitTimer
from vllm import LLM, SamplingParams


TOO_MANY_TOKENS_FOR_ENGINE: str = "This model's maximum context length is"


class PromptOverlengthError(ValueError):
    """
    A ValueError specific to the case where the prompt is longer than the permitted number of tokens
    """
    pass


class OpenAIAPIConfig(TypedDict):
    """
    A dictionary of config items for OpenAI API use
    """
    api_key: str
    organization: Optional[str]  # optional to toggle between a chosen one and API key default
    seconds_per_step: float


def _load_environment_codex_config() -> OpenAIAPIConfig:
    api_key: str = os.environ.get("OPENAI_API_KEY_JLAB_ORG") or os.environ.get("OPENAI_API_KEY")
    organization: str = os.environ.get("OPENAI_ORGANIZATION")
    check_argument(api_key, "must set an API key. Use environment variable OPENAI_API_KEY or otherwise provide "
                            "a CodexConfig")
    return {"api_key": api_key.strip(),  # easier than fixing a k8s secret
            "organization": organization,
            "seconds_per_step": 0.2}


class CodexClient(AbstractLMClient):
    """
    Simplified client for working with Codex and OpenAI models, wraps openai client.
    """

    config: OpenAIAPIConfig
    engine: str
    stop_sequences: List[str]
    timer: SpeedLimitTimer

    def __init__(self, config: OpenAIAPIConfig = None, engine: str = "gpt-3.5-turbo-0125",
                 stop_sequences: List[str] = None, beam_search_config=None) -> None:
        super().__init__()
        self.config = config or _load_environment_codex_config()
        self.engine = engine
        self.stop_sequences = stop_sequences or ['--', '\n', ';', '#']
        self.timer = SpeedLimitTimer(second_per_step=self.config['seconds_per_step'])  # openai limitation 20 query/min
        self.beam_search_config = beam_search_config

    def greedy_lm_completion(self, prompt_text: str) -> Dict[str, float]:
        """
        Given a prompt, generate a completion using the given engine and other completion parameters.
    
        :param prompt_text: prefix text for OpenAI Completion API
        :return: the single most likely completion for the prompt (greedily sampled), not including the prompt tokens.
        """
        stop_sequences = self.stop_sequences or ['--', '\n', ';', '#']
        openai.api_key = self.config['api_key']
        if "organization" in self.config:
            openai.organization = self.config['organization']
        try:
            args: Dict[str, Any] = {
                "model": self.engine,
                "messages": prompt_text,
                "max_tokens": 120,
                "logprobs": True,
                "temperature": 0.0,
                "stop": stop_sequences,
            }
            if self.beam_search_config:
                args.update(self.beam_search_config)
            self.timer.step()
            # result = openai.completions.create(**args)
            result = openai.chat.completions.create(**args)
            completions = dict(zip(
                [x.message.content for x in result.choices],
                [sum(token.logprob for token in x.logprobs.content) for x in result.choices]
            ))
            # completions = dict(zip(
            #     [x.text for x in result.choices],
            #     [sum(x.logprobs.token_logprobs) for x in result.choices]
            # ))
            return completions
        except BadRequestError as e:
            # if e.user_message.startswith(TOO_MANY_TOKENS_FOR_ENGINE):
            #     raise PromptOverlengthError(e)
            # else:
                raise e
        except (RateLimitError, APIError, APIConnectionError, OpenAIError) as e:
            logging.warning(e)
            self.timer.sleep(10)
            raise e

    def top_p_lm_completion(self, prompt_text: str, top_p: float = 0.9, n: int = 5, best_of: int = 10,
                            max_tokens: int = 120, **kwargs) -> Dict[str, float]:
        """
        Given a prompt, generate a completion using the given engine and other completion parameters.

        :param prompt_text: prefix text for OpenAI Completion API
        :return: the single most likely completion for the prompt (greedily sampled), not including the prompt tokens.
        """
        stop_sequences = self.stop_sequences or ['--', '\n', ';', '#']
        openai.api_key = self.config['api_key']
        if "organization" in self.config:
            openai.organization = self.config['organization']
        try:
            # args = {
            #     "model": self.engine,
            #     "prompt": prompt_text,
            #     "max_tokens": max_tokens,
            #     "top_p": top_p,
            #     "stop": stop_sequences,
            #     "n": n,
            #     "logprobs": 1,  # 1 needed to get back log-probabilities at all, in choice['logprobs']['token_logprobs']
            #     "best_of": best_of,
            # }
            args = {
                "model": 'gpt-3.5-turbo-0125',
                "messages": prompt_text,
                "max_tokens": 120,
                "top_p": 0.9,
                "stop": stop_sequences,
                "n": 2,
                "logprobs": True,  # 1 needed to get back log-probabilities at all, in choice['logprobs']['token_logprobs']
            }
            self.timer.step()
            # result = openai.completions.create(**args)
            result = openai.chat.completions.create(**args)
            completions = dict(zip(
                [x.message.content for x in result.choices],
                [sum(token.logprob for token in x.logprobs.content) for x in result.choices]
            ))
            # completions = dict(zip(
            #     [x.text for x in result.choices],
            #     [sum(x.logprobs.token_logprobs) for x in result.choices]
            # ))
            return completions
        except BadRequestError as e:
            # if e.user_message.startswith(TOO_MANY_TOKENS_FOR_ENGINE):
            #   raise PromptOverlengthError(e)
            # else:
            raise e
        except (RateLimitError, APIError, APIConnectionError, OpenAIError) as e:
            logging.warning(e)
            self.timer.sleep(10)
            raise e

    def get_completion_log_probabilities(self, prompt_text: str, completion: str,
                                         token_log_probs_telemetry_hook: Callable[[List[float]], None] = None) -> List[float]:
        stop_sequences = self.stop_sequences or ['--', '\n', ';', '#', ' ']
        openai.api_key = self.config['api_key']
        if "organization" in self.config:
            openai.organization = self.config['organization']
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.engine)
            prompt_length = len(encoding.encode(prompt_text+completion)) + 1
            args = {
                "model": self.engine,
                "prompt": "GIVEN CODE: "+ prompt_text+completion +\
                    " Please echo GIVEN CODE verbatim,\
                      without missing or altering a single character.\n\n",
                "max_tokens": prompt_length,
                "logprobs": 1,
                "temperature": 0.0,
                # "stop": stop_sequences,
                # "echo": False
            }
            self.timer.step()
            result = openai.completions.create(**args)
            # result = openai.chat.completions.create(**args)

            print(f"got log probability for {completion}")
            tokens = result.choices[0].logprobs.tokens
            log_probs = result.choices[0].logprobs.token_logprobs
            if (prompt_text + completion) != "".join(tokens):
                # chop off last one, since we added 1 token in generation
                tokens = tokens[1:]
                log_probs = log_probs[1:]
            # count back to the index of the first token:
            i = len(tokens)
            remaining = completion
            while len(remaining) > 0:
                token = tokens[i - 1]
                i -= 1
                remaining = remaining[:-len(token)]

            # return the log probability of the partial sequence consisting only of the completion
            completion_token_log_probs: List[float] = log_probs[i:-1]
            if token_log_probs_telemetry_hook:
                token_log_probs_telemetry_hook(completion_token_log_probs)
            return completion_token_log_probs

        except BadRequestError as e:
            # if e.user_message.startswith(TOO_MANY_TOKENS_FOR_ENGINE):
            #     raise PromptOverlengthError(e)
            # else:
                raise e
        except (RateLimitError, APIError, APIConnectionError, OpenAIError) as e:
            logging.warning(e)
            self.timer.sleep(10)
            raise e


class LlamaClient(AbstractLMClient):
    """
    Simplified client for working with Codex and OpenAI models, wraps openai client.
    """
    
    engine: str
    stop_sequences: List[str]
    timer: SpeedLimitTimer

    def __init__(self, config = None, engine: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 stop_sequences: List[str] = None, use_vllm: bool = True, quantization: str = None, beam_search_config=None) -> None:
        super().__init__()
        self.config = config
        self.engine = engine
        self.stop_sequences = stop_sequences or ['--', '\n', ';', '#']
        self.use_vllm = use_vllm
        self.timer = SpeedLimitTimer(second_per_step=0.2)  # openai limitation 20 query/min
        self.beam_search_config = beam_search_config

        if use_vllm:
            self.model = LLM(model=self.engine, quantization=quantization, enforce_eager=True)
            self.tokenizer = self.model.get_tokenizer()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.engine)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = LlamaForCausalLM.from_pretrained(
                self.engine,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=False
            )
        self.terminators =  [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")    
        ]

    def greedy_lm_completion(self, prompt_text: str) -> Dict[str, float]:
        """
        Given a prompt, generate a completion using the given engine and other completion parameters.
    
        :param prompt_text: prefix text for OpenAI Completion API
        :return: the single most likely completion for the prompt (greedily sampled), not including the prompt tokens.
        """
        stop_sequences = self.stop_sequences or ['--', '\n', ';', '#']
        
        sequences, logits = [], []
        try:            
            self.timer.step()
            # result = openai.completions.create(**args)
            if self.use_vllm:
                sampling_params = SamplingParams(
                    n=1, best_of=1, max_tokens=120, 
                    temperature=0, stop=stop_sequences,
                    stop_token_ids=self.terminators)
                if self.beam_search_config:
                    sampling_params.use_beam_search = True
                    sampling_params.best_of = self.beam_search_config['beam_size']

                if not isinstance(prompt_text[0], torch.Tensor):
                    prompt_text = [self.tokenizer.apply_chat_template(
                        prompt_text, add_generation_prompt=True, return_tensors="pt"
                    )]
                prompts = [self.tokenizer.batch_decode(prompt, skip_special_tokens=False)[0] for prompt in prompt_text]
                result = self.model.generate(prompts, sampling_params=sampling_params)
                if len(result) > 1:
                    # if batched
                    completions = [{output.outputs[0].text: 1} for output in result]
                else:
                    completions = [{result[0].outputs[0].text: 1}]
            else:
                input_ids = self.tokenizer.apply_chat_template(
                prompt_text,
                add_generation_prompt=True,
                return_tensors="pt"
                )
                input_len = input_ids.shape[-1]
                result = self.model.generate(
                    input_ids.to(self.model.device),
                    max_new_tokens=120,
                    eos_token_id=self.terminators,
                    do_sample=False,
                    output_scores= True,
                    return_dict_in_generate=True,
                    stop_strings=stop_sequences,
                    tokenizer=self.tokenizer,
                    output_logits=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                sequences.append(result['sequences'][0])
                logits.append(result['logits'])

                completions = dict(zip(
                    [self.tokenizer.decode(seq[input_len:], skip_special_tokens=True) for seq in sequences] ,
                    [torch.stack(logit).softmax(dim=-1).max(dim=-1)[0].log().sum().item() for logit in logits]))
            return completions
        except BadRequestError as e:
            # if e.user_message.startswith(TOO_MANY_TOKENS_FOR_ENGINE):
            #     raise PromptOverlengthError(e)
            # else:
                raise e
        except (RateLimitError, APIError, APIConnectionError, OpenAIError) as e:
            logging.warning(e)
            self.timer.sleep(10)
            raise e
    def top_p_lm_completion(self, prompt_text: str, top_p: float = 0.9, n: int = 5, best_of: int = 10,
                            max_tokens: int = 120, **kwargs) -> Dict[str, float]:
        """
        Given a prompt, generate a completion using the given engine and other completion parameters.

        :param prompt_text: prefix text for OpenAI Completion API
        :return: the single most likely completion for the prompt (greedily sampled), not including the prompt tokens.
        """
        stop_sequences = self.stop_sequences or ['--', '\n', ';', '#']
        terminators =  [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")    
        ]
        input_ids = self.tokenizer.apply_chat_template(
                prompt_text,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
        input_len = input_ids.shape[-1]
        sequences, logits = [], []

        try:
            
            for _ in range(2):
                # result = openai.completions.create(**args)
                result = self.model.generate(
                    input_ids,
                    max_new_tokens=120,
                    eos_token_id=self.terminators,
                    do_sample=True,
                    top_p=0.9,
                    output_scores= True,
                    return_dict_in_generate=True,
                    stop_strings=stop_sequences,
                    tokenizer=self.tokenizer,
                    output_logits=True,
                    )

                sequences.append(result['sequences'][0])
                logits.append(result['logits'])
            completions = dict(zip(
                [self.tokenizer.decode(seq[input_len:], skip_special_tokens=True) for seq in sequences] ,
                [torch.stack(logit).softmax(dim=-1).max(dim=-1)[0].log().sum().item() for logit in logits]))
        
            return completions
        except BadRequestError as e:
            # if e.user_message.startswith(TOO_MANY_TOKENS_FOR_ENGINE):
            #   raise PromptOverlengthError(e)
            # else:
            raise e
        except (RateLimitError, APIError, APIConnectionError, OpenAIError) as e:
            logging.warning(e)
            self.timer.sleep(10)
            raise e

    def get_completion_log_probabilities(self, prompt_text: str, completion: str,
                                         token_log_probs_telemetry_hook: Callable[[List[float]], None] = None) -> List[float]:
        stop_sequences = self.stop_sequences or ['--', '\n', ';', '#', ' ']
        openai.api_key = self.config['api_key']
        if "organization" in self.config:
            openai.organization = self.config['organization']
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.engine)
            prompt_length = len(encoding.encode(prompt_text+completion)) + 1
            args = {
                "model": self.engine,
                "prompt": "GIVEN CODE: "+ prompt_text+completion +\
                    " Please echo GIVEN CODE verbatim,\
                      without missing or altering a single character.\n\n",
                "max_tokens": prompt_length,
                "logprobs": 1,
                "temperature": 0.0,
                # "stop": stop_sequences,
                # "echo": False
            }
            self.timer.step()
            result = openai.completions.create(**args)
            # result = openai.chat.completions.create(**args)

            print(f"got log probability for {completion}")
            tokens = result.choices[0].logprobs.tokens
            log_probs = result.choices[0].logprobs.token_logprobs
            if (prompt_text + completion) != "".join(tokens):
                # chop off last one, since we added 1 token in generation
                tokens = tokens[1:]
                log_probs = log_probs[1:]
            # count back to the index of the first token:
            i = len(tokens)
            remaining = completion
            while len(remaining) > 0:
                token = tokens[i - 1]
                i -= 1
                remaining = remaining[:-len(token)]

            # return the log probability of the partial sequence consisting only of the completion
            completion_token_log_probs: List[float] = log_probs[i:-1]
            if token_log_probs_telemetry_hook:
                token_log_probs_telemetry_hook(completion_token_log_probs)
            return completion_token_log_probs

        except BadRequestError as e:
            # if e.user_message.startswith(TOO_MANY_TOKENS_FOR_ENGINE):
            #     raise PromptOverlengthError(e)
            # else:
                raise e
        except (RateLimitError, APIError, APIConnectionError, OpenAIError) as e:
            logging.warning(e)
            self.timer.sleep(10)
            raise e
