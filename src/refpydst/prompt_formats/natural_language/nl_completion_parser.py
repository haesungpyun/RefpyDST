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