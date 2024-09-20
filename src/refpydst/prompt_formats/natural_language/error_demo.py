import copy
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, get_type_hints, Optional, Final, Any

from refpydst.data_types import SlotValue, MultiWOZDict, SlotValuesByDomain

from refpydst.db.ontology import CATEGORICAL_SLOT_VALUE_TYPES
from refpydst.prompt_formats.python.python_classes import Hotel, Train, Attraction, Restaurant, Taxi, Option
from refpydst.utils.dialogue_state import compute_delta

import random
from refpydst.db.ontology import Ontology

def get_erroneous_prediction(current_bs: MultiWOZDict, previous_bs)->Tuple[str, MultiWOZDict]:
    """
    Given a belief state, return an erroneous prediction
    """
    ontology = Ontology.create_ontology()
    if len(current_bs) == 0:
        error_case = 'hall'
    else:
        error_case = random.choice(['miss', 'hall'])
    
    
    if error_case == 'miss':
        if '[DELETE]' in list(current_bs.values()) and 'dontcare' in list(current_bs.values()):
            sub_error_case = random.choice(['total', 'confuse', 'delete', 'dontcare'])
        
        elif '[DELETE]' in list(current_bs.values()):
            sub_error_case = random.choice(['total', 'confuse', 'delete'])
        
        elif 'dontcare' in list(current_bs.values()):
            sub_error_case = random.choice(['total', 'confuse', 'dontcare'])
        
        else:
            sub_error_case = random.choice(['total', 'confuse'])
            
        missed_slot = random.choice(list(current_bs.keys()))
        erroneous_prediction = copy.deepcopy(current_bs)
        
        if sub_error_case == 'confuse':
            confused_slot = random.choice(list(ontology.known_values.keys()))
            erroneous_prediction[confused_slot] = current_bs[missed_slot]
            instruction = "---\n## Instructions to avoid **Slot Confusion Error**\n"
            instruction += "### Domain Shifts & Context:\n"
            instruction += "   - **Analyze Each Utterance:** Don't assume continuity; look for keywords signaling a new domain.\n"
            instruction += "   - **Adapt to Intent:** Switch domains based on the current topic, avoiding bias from previous turns.\n\n"
            instruction += "### Slot Assignment:\n"
            instruction += "   - **Assign & Verify:** Ensure slots match the correct domain and intent; double-check for accuracy.\n\n"
            instruction += "### Focus & Flexibility:\n"
            instruction += "   - **Stay Alert & Flexible:** Monitor changes in user needs and shift domains accordingly.\n"
            instruction += "   - **Avoid Carryover:** Don't misassign slots from previous domains.\n---\n"
                

        elif sub_error_case == 'delete':
            missed_slot = [s for s, v in current_bs.items() if v == '[DELETE]'][0]
            
            

        elif sub_error_case == 'dontcare':
            missed_slot = [s for s, v in current_bs.items() if v == 'dontcare'][0]
                        

        else:
            instruction = "---\n## Instructions to avoid **Missing Slot-Value Pairs**\n"
            instruction += "### Consider Explicit and Implicit Information:\n"
            instruction +="   - **Infer Implicit Acceptance:** Don't rely only on explicit input; infer slot-value pairs from positive responses or user actions (e.g., proceeding with booking details).\n\n"
            instruction +="### System Utterances:\n"
            instruction +="   - **Incorporate System Information:** Extract relevant slot-value pairs from system suggestions, especially when the user implicitly accepts them.\n\n"
            instruction +="### Dialogue Context:\n"
            instruction +="   - **Leverage Full Context:** Use the entire conversation history to understand intent, capturing implicit confirmations or denials for accurate belief state updates.\n---\n"
            
        del erroneous_prediction[missed_slot]
        return f'{error_case}_{sub_error_case}', instruction, erroneous_prediction   
    else:
        if previous_bs is not None or len(previous_bs) > 0:
            sub_error_case = random.choice(['total', 'val', 'overwrite'])
        else:
            sub_error_case = random.choice(['total', 'val'])
        

        if sub_error_case == 'total':
            pass


    return erroneous_prediction