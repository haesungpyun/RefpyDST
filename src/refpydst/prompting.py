from typing import List, Dict, Literal

from refpydst.data_types import Turn, CompletionParser, MultiWOZDict

import refpydst.prompt_formats.python.demo as python_demo
from refpydst.prompt_formats.python.completion_parser import parse_python_completion
from refpydst.prompt_formats.utils import promptify_slot_names
from refpydst.resources import _read_resource
from refpydst.utils.dialogue_state import compute_delta
from refpydst.utils.general import read_json_from_data_dir
from refpydst.utils.sql import slot_values_to_seq_sql, sql_pred_parse

IC_DST: str = "IC-DST"
IC_DST_NULL: str = "IC-DST-NULL"
PYTHON_PROMPT: str = "python-prompt"
PYTHON_PROMPT_NULL: str = "python-prompt-NULL"
PROMPT_VARIANTS: List[str] = [IC_DST, IC_DST_NULL, PYTHON_PROMPT, PYTHON_PROMPT_NULL]


STOP_SEQUENCES: Dict[str, List[str]] = {
    PYTHON_PROMPT: ["\n\n", "#", "print("],
    IC_DST: ['--', '\n', ';', '#']
}


def default_sql_completion_parser(completion: str, _: MultiWOZDict, **kwargs) -> MultiWOZDict:
    # convert back the sql completion result
    completion = promptify_slot_names(completion, reverse=True)
    return sql_pred_parse(completion)

SlotName = Literal[
    "attraction-area", "attraction-name", "attraction-type", 
    "bus-day", "bus-departure","bus-destination", "bus-leaveat", "hospital-department", 
    "hotel-area", "hotel-book day", "hotel-book people", "hotel-book stay", "hotel-internet", "hotel-name", "hotel-parking", "hotel-pricerange", "hotel-stars", "hotel-type", 
    "restaurant-area", "restaurant-book day", "restaurant-book people", "restaurant-book time", "restaurant-food", "restaurant-name", "restaurant-pricerange", 
    "taxi-arriveby", "taxi-departure", "taxi-destination", "taxi-leaveat",
    "train-arriveby", "train-book people", "train-day", "train-departure", "train-destination", "train-leaveat"
]
def parse_nl_completion(nl_completion: str, state = None,
                            exceptions_are_empty: bool = True, **kwargs) -> MultiWOZDict:
    """
    The dialogue state change due to the lastest turn is like this:
    The value of slot book time of restaurant is 11:15.
    
     Parses a python completion to a complete dialogue state for the turn, y_t.

    :param python_completion: the dialogue state update in python function-call form
    :param state: the existing dialogue state (y_{t-1})
    :param exceptions_are_empty: If an exception is encountered (un-parseable completion), treat this as containing no
      update to the dialogue state.
    :param kwargs: Not used, but included as other parsers use different arguments.
    :return: the updated dialogue state y_t.

    {'attraction-name': 'cineworld cinema', 'train-book people': '6', 'train-destination': 'cambridge', 'train-day': 'wednesday', 'train-arriveby': '19:45', 'train-departure': 'london kings cross'}

    """
    try:
        full_statement = nl_completion.strip()
        full_statement = full_statement.replace('{', '').replace('}', '')
        
        bs_dict = {}
        for d_s_v in full_statement.split(','):
            d_s = eval(d_s_v.split(":")[0])
            v = d_s_v.split(":")[1:]
            v = str(eval(":".join(v)))
            try:
                assert f"{d_s}" in SlotName.__args__
            except AssertionError:
                print(f"{d_s} not found in SlotName")
                continue
            bs_dict[d_s] = v

        return bs_dict
    except Exception as e:
        # print(f"got exception when parsing: {pprint.pformat(e)}")
        # logging.warning(e)
        # if not exceptions_are_empty:
        #     raise e
        return bs_dict


def get_completion_parser(prompt_format: str) -> CompletionParser:
    if prompt_format == PYTHON_PROMPT:
        return parse_python_completion
    elif prompt_format == IC_DST:
        return default_sql_completion_parser
    else:
        return parse_nl_completion


class PromptGenerator:
    """
    A class handling the creation of prompts for various experiments
    """
    preambles: Dict[str, str]

    def __init__(self):
        ic_dst_table_prompt: str = _read_resource("prompt_formats/ic_dst/table.sql")
        self.preambles = {
            IC_DST: ic_dst_table_prompt,
            IC_DST_NULL: ic_dst_table_prompt,
            PYTHON_PROMPT: _read_resource("prompt_formats/python/python_classes.py"),
            PYTHON_PROMPT_NULL: _read_resource("prompt_formats/python/python_classes.py"),
        }

    def get_prompt(
        self, 
        data_item, examples, given_context=None, n_examples=None, 
        prompt_format: str = None, chat_format: bool=False, add_guidelines: bool = True):
        """
        You can try different prompt in here.
        """
        # Note the IC-DST text-to-sql prompts are all prefixed with "IC-DST":
        if not prompt_format or prompt_format in (IC_DST, IC_DST_NULL):
            if prompt_format == IC_DST_NULL:
                reverse_x_and_y, use_null_data_item = True, True
            else:
                reverse_x_and_y, use_null_data_item = False, False

            if not chat_format:
                final_prompt = self.get_icdst_prompt(
                    data_item, examples, given_context=given_context, n_examples=n_examples,
                    reverse_x_and_y=reverse_x_and_y, use_null_data_item=use_null_data_item, add_guidelines=add_guidelines
                )
            else:
                final_prompt = self.get_icdst_chat_prompt(
                    data_item, examples, given_context=given_context, n_examples=n_examples,
                    reverse_x_and_y=reverse_x_and_y, use_null_data_item=use_null_data_item, add_guidelines=add_guidelines
                )
            # return final_prompt
        elif prompt_format in (PYTHON_PROMPT, PYTHON_PROMPT_NULL):
            if prompt_format == PYTHON_PROMPT_NULL:
                reverse_x_and_y, use_null_data_item = True, True
            else:
                reverse_x_and_y, use_null_data_item = False, False
            detailed_state_string: bool = prompt_format == PYTHON_PROMPT
            if not chat_format:
                final_prompt = self.get_python_prompt(
                    data_item, examples, given_context=given_context, n_examples=n_examples, 
                    reverse_x_and_y=reverse_x_and_y, use_null_data_item=use_null_data_item, 
                    detailed_state_string=detailed_state_string, add_guidelines=add_guidelines
                )
            else:
                final_prompt = self.get_python_chat_prompt(
                    data_item, examples, given_context=given_context, n_examples=n_examples,
                    reverse_x_and_y=reverse_x_and_y, use_null_data_item=use_null_data_item, 
                    detailed_state_string=detailed_state_string, add_guidelines=add_guidelines
                )
            # return final_prompt
        
        else:
            if not chat_format:
                raise ValueError(f"Prompt with Plain_text only support chat template. Set chat_format=True")
            else:
                final_prompt = self.get_nl_chat_prompt(
                    data_item, examples, given_context=given_context, n_examples=n_examples,
                    reverse_x_and_y=False, use_null_data_item=False, detailed_state_string=True, add_guidelines=add_guidelines
                )
        return final_prompt

    @staticmethod
    def get_canonical_completion(slot_values: MultiWOZDict, context_slot_values, turn: Turn,
                                 prompt_format: str = None):
        """
        For a given true value y or prediction y_hat, generate a string that the LM could have completed given the
        relevant prompt in order to produce y/y_hat when parsed
        """
        # Note the IC-DST text-to-sql prompts are all prefixed with "IC-DST":
        slot_delta = compute_delta(context_slot_values, slot_values)
        if not prompt_format or prompt_format.startswith(IC_DST):
            # chop off the end, as we complete it in the prompts
            return f"{promptify_slot_names(slot_values_to_seq_sql(slot_delta))}".replace("SELECT * FROM", "")
        elif prompt_format in (PYTHON_PROMPT, PYTHON_PROMPT_NULL):
            last_sys_utt = turn['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            user_string = python_demo.get_user_string(turn['dialog']['usr'][-1])
            _, update_string = python_demo.get_python_statements(context_slot_values, slot_values,
                                                                 turn_strings=[last_sys_utt, user_string])
            if update_string.startswith("agent.state."):
                return update_string.replace("agent.state.", "", 1)
            return update_string
        else:
            raise ValueError(f"Unsupported prompt format: {prompt_format}")

    def get_icdst_prompt(self, data_item, examples, given_context=None, n_examples=None, add_context: bool = True,
                         reverse_x_and_y: bool = False, use_null_data_item: bool = False):
        """
        Prompt as originally proposed in the IC-DST paper
        """
        table_prompt = self.preambles[IC_DST]

        question_item = data_item

        prompt_text = f"{promptify_slot_names(table_prompt)}\n"

        max_n_examples = len(examples)
        if n_examples is not None:
            max_n_examples = n_examples

        # in case for zero-shot learning
        if max_n_examples > 0:
            for example_id, example in enumerate(examples[-max_n_examples:]):
                turn_text = f"Example #{example_id + 1}\n"
                turn_input_text = ""
                # remove multiple choice in last slot values
                if add_context:
                    last_slot_values = {s: v.split('|')[0] for s, v in example['last_slot_values'].items()}
                    turn_input_text += f"[context] {promptify_slot_names(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"

                last_sys_utt = example['dialog']['sys'][-1]
                if last_sys_utt == 'none':
                    last_sys_utt = ''
                turn_input_text += f"[system] {last_sys_utt}\n"
                turn_input_text += f"Q: [user] {example['dialog']['usr'][-1]}\n"

                turn_output_text = f"SQL: {promptify_slot_names(slot_values_to_seq_sql(example['turn_slot_values']))};\n"

                # set the text for this turn, depending on order preference
                turn_text += (turn_input_text + turn_output_text) if not reverse_x_and_y else \
                    (turn_output_text + turn_input_text)
                prompt_text += turn_text + "\n\n"

        prompt_text += f"Example #{max_n_examples + 1}\n"
        if given_context is None:
            last_slot_values = {s: v.split(
                '|')[0] for s, v in question_item['last_slot_values'].items()}
        else:
            last_slot_values = given_context
        test_example_text: str = ""
        if add_context:
            test_example_text += f"[context] {promptify_slot_names(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"

        last_sys_utt = question_item['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        test_example_text += f"[system] {last_sys_utt}\n"
        test_example_text += f"Q: [user] {question_item['dialog']['usr'][-1]}\n"
        test_example_text += "SQL: SELECT * FROM"

        if not use_null_data_item:
            prompt_text += test_example_text
        else:
            # use a null input (note for now we have not chosen a leading null X -> Y input)
            prompt_text += ("SQL: SELECT * FROM" if reverse_x_and_y else "")
        return prompt_text

    def get_icdst_chat_prompt(self, data_item, examples, given_context=None, n_examples=None, add_context: bool = True,
                         reverse_x_and_y: bool = False, use_null_data_item: bool = False, add_guidelines:bool = True):
        """
        Prompt as originally proposed in the IC-DST paper
        """
        msg = [{"role": "system", "content": "You are an expert in Dialogue State Tracking(DST) and SQL coding. DO NOT say anything except SQL code.\n"}]
        table_prompt = self.preambles[IC_DST]

        question_item = data_item

        prompt_text = f"{promptify_slot_names(table_prompt)}\n"

        max_n_examples = len(examples)
        if n_examples is not None:
            max_n_examples = n_examples

        # in case for zero-shot learning
        if max_n_examples > 0:
            for example_id, example in enumerate(examples[-max_n_examples:]):
                prefix_msg = f"Example #{example_id + 1}\n"
                user_string = ""
                
                # remove multiple choice in last slot values
                if add_context:
                    last_slot_values = {s: v.split('|')[0] for s, v in example['last_slot_values'].items()}
                    user_string += f"[context] {promptify_slot_names(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"

                last_sys_utt = example['dialog']['sys'][-1]
                if last_sys_utt == 'none':
                    last_sys_utt = ''
                user_string += f"[system] {last_sys_utt}\n"
                user_string += f"Q: [user] {example['dialog']['usr'][-1]}\n"

                state_string = f"SQL: {promptify_slot_names(slot_values_to_seq_sql(example['turn_slot_values']))};\n"

                # set the text for this turn, depending on order preference
                if not reverse_x_and_y:
                    msg.append({"role": "user","content": prefix_msg+user_string})
                    state_string += "\n\n"
                    msg.append({"role": "assistant","content": state_string})
                else:
                    msg.append({"role": "user","content": prefix_msg+state_string})
                    user_string += "\n\n"
                    msg.append({"role": "assistant","content": user_string})

        prefix_msg = f"Example #{max_n_examples + 1}\n"
        if given_context is None:
            last_slot_values = {s: v.split('|')[0] for s, v in question_item['last_slot_values'].items()}
        else:
            last_slot_values = given_context
        user_string: str = ""
        if add_context:
            user_string += f"[context] {promptify_slot_names(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"

        last_sys_utt = question_item['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        user_string += f"[system] {last_sys_utt}\n"
        user_string += f"Q: [user] {question_item['dialog']['usr'][-1]}\n"
        state_string += "SQL: SELECT * FROM"

        if not use_null_data_item:
            msg.append({"role": "user","content": prefix_msg+user_string})
            msg.append({"role": "assistant","content": state_string})
        else:
            # use a null input (note for now we have not chosen a leading null X -> Y input)
            msg.append({"role": "user","content": prefix_msg})
            if reverse_x_and_y:
                msg.append({"role":"assistant", "content":state_string})
            else:
                msg.append({"role": "assistant","content": ""})
        if add_guidelines:
            msg[1]['content'] = prompt_text + msg[1]['content']
        return msg


    def get_python_prompt(self, data_item, examples, given_context=None, n_examples: int = None,
                          reverse_x_and_y: bool = False, use_null_data_item: bool = False,
                          detailed_state_string: bool = False) -> str:
        lines: List[str] = []
        max_n_examples: int = n_examples is not None and n_examples or len(examples)

        # in case for zero-shot learning
        if max_n_examples > 0:
            for example_id, example in enumerate(examples[-max_n_examples:]):
                lines.append(f"# Example {example_id + 1}")
                turn_inputs, turn_outputs = [], []

                # remove multiple choice in last slot values
                last_slot_values = {s: v.split('|')[0] for s, v in example['last_slot_values'].items()}

                last_sys_utt = example['dialog']['sys'][-1]
                if last_sys_utt == 'none':
                    last_sys_utt = ''
                user_string = python_demo.get_user_string(example['dialog']['usr'][-1])
                state_string, update_string = python_demo.get_python_statements(last_slot_values, example['slot_values'],
                                                                                turn_strings=[last_sys_utt, user_string],
                                                                                detailed_state_string=detailed_state_string)
                turn_inputs.append(state_string)
                if last_sys_utt:
                    turn_inputs.append(python_demo.get_system_string(last_sys_utt))
                turn_inputs.append(user_string)
                turn_outputs.extend([s.strip() for s in update_string.split("\n")])
                if not reverse_x_and_y:
                    lines.extend(turn_inputs)
                    lines.extend(turn_outputs)
                else:
                    lines.extend(turn_outputs)
                    lines.extend(turn_inputs)
                lines.append("")

        lines.append(f"# Example {max_n_examples + 1}")
        if given_context is None:
            last_slot_values = {s: v.split(
                '|')[0] for s, v in data_item['last_slot_values'].items()}
        else:
            last_slot_values = given_context
        last_sys_utt = data_item['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        user_string = python_demo.get_user_string(data_item['dialog']['usr'][-1])
        state_string, _ = python_demo.get_python_statements(last_slot_values, {},
                                                            turn_strings=[last_sys_utt, user_string],
                                                            detailed_state_string=detailed_state_string)
        if not use_null_data_item:
            lines.append(state_string)
            if last_sys_utt:
                lines.append(python_demo.get_system_string(last_sys_utt))
            lines.append(user_string)
        else:
            pass  # default adds our null input at end
        prompt_text = self.preambles[PYTHON_PROMPT] + "    " + "\n    ".join(lines) + "\n    agent.state."
        return prompt_text

    def get_python_chat_prompt(self, data_item, examples, given_context=None, n_examples: int = None,
                          reverse_x_and_y: bool = False, use_null_data_item: bool = False,
                          detailed_state_string: bool = True, add_guidelines:bool = True) -> str:
    
        msg = [{"role": "system", "content": "You are an expert in Dialogue State Tracking(DST) and python coding.\n"}]
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
        # _, gt_string = python_demo.get_python_statements(last_slot_values, data_item['slot_values'],
        #                                                     turn_strings=[last_sys_utt, user_string],
        #                                                     detailed_state_string=detailed_state_string)
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
        if add_guidelines:
            msg[1]['content'] = self.preambles[PYTHON_PROMPT] + "    \n" + msg[1]['content']
        return msg

        
    def get_nl_chat_prompt(self, data_item, examples, given_context=None, n_examples: int = None,
                            reverse_x_and_y: bool = False, use_null_data_item: bool = False,
                            detailed_state_string: bool = True, add_guidelines:bool = True) -> str:
        system_msg = "**Task:** You are an expert in Dialogue State Tracking (DST) focused on managing and updating the dialogue state change based on system-user interactions. "
        system_msg += "The dialogue state represents the user's preferences and booking details across different domains: Hotel, Train, Attraction, Restaurant, and Taxi.\n\n"
        msg = [{"role": "system", "content": system_msg}]
        max_n_examples: int = n_examples is not None and n_examples or len(examples)

        # in case for zero-shot learning
        if max_n_examples > 0:
            for example_id, example in enumerate(examples[-max_n_examples:]):
                prefix_msg = f"\n**Example {example_id + 1} of Dialogue State Change Update Task:**\n"
                
                # remove multiple choice in last slot values
                last_slot_values = {s: v.split('|')[0] for s, v in example['last_slot_values'].items()}
                turn_slot_values = {s: v.split('|')[0] for s, v in example['turn_slot_values'].items()}

                last_sys_utt = example['dialog']['sys'][-1]
                if last_sys_utt == 'none':
                    last_sys_utt = ''
                
                state_string = '    **Previous Belief State (Before the Latest User Interaction):** \n'
                # for s, v in last_slot_values.items():
                #     state_string += f"    {{**{s.split('-')[0]}**-{s.split('-')[1]}\": \"{v}\"}}"
                state_string += f"        {last_slot_values}\n\n"
                
                turn_msg = state_string + "    **Latest Conversation Between System and User:** \n"
                if last_sys_utt:
                    turn_msg += '        **System:** "' + last_sys_utt + '"\n'
                
                turn_msg += '        **User:** "' + example['dialog']['usr'][-1] + '"\n\n'
                
                turn_msg += '    **Instructions:**\n'
                turn_msg += '        - Based on the user\'s latest input, update the belief state by correctly identifying and filling in the relevant domain(s), slot(s) and value(s).\n'
                turn_msg += '        - Provide your output strictly in the Required Output Format below.\n\n'
                turn_msg += "    **Required Output Format:**\n"
                turn_msg += "        **Dialogue state change after Latest Conversation Between System and User:** \n"
                # for s, v in turn_slot_values.items():
                #     bs_msg += f"The value of slot \"{s.split('-')[1]}\" of \"{s.split('-')[0]}\" is {v}. "
                bs_msg = f"            {turn_slot_values}\n"
                if not reverse_x_and_y:
                    msg.append({"role": "user","content": prefix_msg+turn_msg})
                    msg.append({"role": "assistant","content": bs_msg})
                else:
                    msg.append({"role": "user", "content": prefix_msg+bs_msg})
                    msg.append({"role": "assistant","content": turn_msg})

        prefix_msg = f"\n**Example {max_n_examples + 1} of Dialogue State Change Update Task:**\n"
        if given_context is None:
            last_slot_values = {s: v.split('|')[0] for s, v in data_item['last_slot_values'].items()}
        else:
            last_slot_values = given_context
        
        last_sys_utt = data_item['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        
        state_string = '    **Previous Belief State (Before the Latest User Interaction):** \n'
        # for s, v in last_slot_values.items():
        #     state_string += f"    {{**{s.split('-')[0]}**-{s.split('-')[1]}\": \"{v}\"}}"
        state_string += f"        {last_slot_values}\n\n"

        turn_msg = ''
        if not use_null_data_item:
            turn_msg = state_string + "    **Latest Conversation Between System and User:** \n"
            if last_sys_utt:
                    turn_msg += '        **System:** "' + last_sys_utt + '"\n'
            turn_msg += '        **User:** "' + data_item['dialog']['usr'][-1] + '"\n\n'
            turn_msg += '    **Instructions:**\n'
            turn_msg += '        - Based on the user\'s latest input, update the belief state by correctly identifying and filling in the relevant domain(s), slot(s) and value(s).\n'
            turn_msg += '        - Provide your output strictly in the Required Output Format below.\n\n'
            turn_msg += "    **Required Output Format:**\n"
            turn_msg += "        **Dialogue state change after Latest Conversation Between System and User:** \n"
        else:
            pass  # default adds our null input at end
        msg.append({"role": "user","content": prefix_msg+turn_msg})
        # msg.append({"role": "assistant","content": "    agent.state."})
        if add_guidelines:
            tmp_msg_content = msg[1]['content']
            msg[1]['content'] = '**Guidelines:**\n\n'
            msg[1]['content'] +='    1. **Hotel:**\n'
            msg[1]['content'] +='        - **Slots:** name (string), pricerange (PriceRange), type (HotelType), parking (Option), book stay (integer), book day (DayOfWeek), book people (integer), area (Area), stars (integer between 0 and 5 or "dontcare"), internet (Option).\n'
            msg[1]['content'] +='        - **Valid Values:**\n'
            msg[1]['content'] +='            - PriceRange: "dontcare", "cheap", "moderate", "expensive".\n'
            msg[1]['content'] +='            - HotelType: "hotel", "guest house", "dontcare".\n'
            msg[1]['content'] +='            - Option: "yes", "no", "dontcare".\n'
            msg[1]['content'] +='            - DayOfWeek: "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday".\n'
            msg[1]['content'] +='            - Area: "dontcare", "centre", "east", "north", "south", "west".\n\n'
            msg[1]['content'] +='    2. **Train:**\n'
            msg[1]['content'] +='        - **Slots:** destination (string), departure (string), day (DayOfWeek), book people (integer), leaveat (hh:mm or "dontcare"), arriveby (hh:mm or "dontcare").\n\n'
            msg[1]['content'] +='    3. **Attraction:**\n'
            msg[1]['content'] +='        - **Slots:** name (string), area (Area), type (AttractionType).\n'
            msg[1]['content'] +='        - **Valid Values:**\n'
            msg[1]['content'] +='            - AttractionType: "architecture", "boat", "church", "cinema", "college", "concert hall", "entertainment", "hotspot", "multiple sports", "museum", "nightclub", "park", "special", "swimming pool", "theatre", "dontcare".\n\n'
            msg[1]['content'] +='    4. **Restaurant:**\n'
            msg[1]['content'] +='        - **Slots:** name (string), food (string), pricerange (PriceRange), area (Area), book time (hh:mm or "dontcare"), book day (DayOfWeek), book people (integer).\n\n'
            msg[1]['content'] +='    5. **Taxi:**\n'
            msg[1]['content'] +='        - **Slots:** destination (string), departure (string), leaveat (hh:mm or "dontcare"), arriveby (hh:mm or "dontcare").\n\n' 
            msg[1]["content"] += tmp_msg_content
        return msg


if __name__ == '__main__': 
    import os 
    os.environ['OPENAI_API_KEY'] = "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"
    os.environ['REFPYDST_DATA_DIR'] = "/home/haesungpyun/RefPyDST/data"
    os.environ['REFPYDST_OUTPUTS_DIR'] = "/home/haesungpyun/RefPyDST/outputs"
    
    pg = PromptGenerator()
    data = read_json_from_data_dir("mw24_10p_dev.json")
    train_data = read_json_from_data_dir("mw21_5p_train_v2.json")
    DEMONSTRATION_EXAMPLES = [
        {
            "dialog": {
                "sys": ["i have booked that for you , is there anything else i can help with ?"],
                "usr": [
                    "thank you , i will need a taxi from clare college to the restaurant . i need to get there by reservation time .",
                ]
            },
            "turn_slot_values": {
                "taxi-departure": "clare college",
                "taxi-destination": "restaurant alimentum",
                "taxi-arriveby": "6:15",
            },
            "last_slot_values": {"restaurant-name": "restaurant alimentum", "restaurant-food": "modern european",
                                 "restaurant-book time": "6:15", "restaurant-book people": "2",
                                 "restaurant-book day": "tuesday"},
            "slot_values": {"restaurant-name": "restaurant alimentum", "restaurant-food": "modern european",
                                 "restaurant-book time": "6:15", "restaurant-book people": "2",
                                 "restaurant-book day": "tuesday","taxi-departure": "clare college",
                "taxi-destination": "restaurant alimentum",
                "taxi-arriveby": "6:15",}
        }
    ]
    prompt = pg.get_prompt(data[410], data[306:307], n_examples=1, given_context=data[410]['last_slot_values'],
                           prompt_format=PYTHON_PROMPT,chat_format=True)
    print(prompt)
