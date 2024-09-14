import abc
import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Union, Literal
import copy

from auto_error_analysis.evaluate_run_log import evaluate_logs
from auto_error_analysis.evaluate_metrics import evaluate, slot_level_f1
from auto_error_analysis.normalization.data_ontology_normalizer import DataOntologyNormalizer
from auto_error_analysis.db.ontology import Ontology
from auto_error_analysis.data_types import SlotName, SlotValue, MultiWOZDict
from auto_error_analysis.utils import validate_path_and_make_abs_path,read_json, save_analyzed_log, load_analyzed_log
from auto_error_analysis.bs_utils import compute_dict_difference, sort_data_item, unroll_or, update_dialogue_state
from auto_error_analysis.completion_parser import PARSING_FUNCTIONS

class AbstractAnalyzer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def categorize_error_case(self, pred_bs, gold_bs) -> Dict[SlotName, SlotValue]:
        return NotImplementedError('')

    @abc.abstractmethod
    def analyze(self, pred_bs, gold_bs) -> Dict[SlotName, SlotValue]:
        return NotImplementedError('')

class ErrorAnalyzer(AbstractAnalyzer):
    def __init__(
            self,
            train_data_path: str = None,
            result_file_path: str = None, 
            output_dir_path: str = './',
            ontology_path: str = './src/refpydst/db/multiwoz/2.4/ontology.json',
            parsing_func: str = 'iterative_parsing',
            special_values: List[str] = ['dontcare', '[DELETE]'],
            use_llm: bool = False
    ):  
        """
        Initializes the error analyzer.
            
        Args:
            train_data_path (str): The path to the training data.
            result_file_path (str): The path to the result file.
            output_dir_path (str): The path to save the analyzed log.
            ontology_path (str): The path to the ontology file.
            parsing_func (function): The parsing function to use for iterative parsing.
                                     [iterative_parsing(default), parse_python_modified, parse_state_change, parse_python_completion]
            special_values (List[str]): The special values to consider as errors.
        """
        
        train_data = read_json(train_data_path)
        self.normalizer = self.get_normalizer(train_data, ontology_path)

        self.result_file_path = validate_path_and_make_abs_path(result_file_path)
        self.output_dir_path = output_dir_path or result_file_path
        self.output_dir_path = validate_path_and_make_abs_path(self.output_dir_path, is_output_dir=True)

        try:
            self.parsing_func = PARSING_FUNCTIONS[parsing_func]
        except KeyError:
            raise ValueError(f"Invalid parsing function: {parsing_func}. Choose from {list(PARSING_FUNCTIONS.keys())}")
        
        self.special_values = special_values or ['dontcare', '[DELETE]']

        self.use_llm = use_llm
    
    def get_normalizer(self, train_data, ontology_path):
        
        ontology_path = validate_path_and_make_abs_path(ontology_path)
        return DataOntologyNormalizer(
                Ontology.create_ontology(),
                # count labels from the train set
                supervised_set=train_data,
                # make use of existing surface form knowledge encoded in ontology.json, released with each dataset
                # see README.json within https://github.com/smartyfh/MultiWOZ2.4/raw/main/data/MULTIWOZ2.4.zip
                counts_from_ontology_file=ontology_path
        )

    def record_error_and_update_visited(
        self,
        error_dict: Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]],
        error_name: str,
        error_s_v_pairs: Union[Tuple[SlotName, SlotValue], Tuple[SlotName, SlotValue, SlotName, SlotValue]],
        visited_pairs: List[Tuple[SlotName, SlotValue]] = []
    ) -> Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]:
        """
        Records an error in the error dictionary and updates the list of visited pairs.

        Args:
            error_dict (Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]]): The error dictionary.
            error_name (str): The name of the error.
            error_s_v_pairs (Union[Tuple[SlotName, SlotValue], Tuple[SlotName, SlotValue, SlotName, SlotValue]]): The slot and value pair(s) associated with the error.
            visited_pairs (List[Tuple[SlotName, SlotValue]]): List of visited slot and value pairs

        Returns:
            Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]: Updated error dictionary and visited pairs list.
        """
        # Append the error to the error dictionary
        if error_name is None or error_s_v_pairs is None:
            return error_dict, visited_pairs
        if (error_name, error_s_v_pairs) in error_dict.get('error', []):
            return error_dict, visited_pairs
    
        error_dict.setdefault('error', []).append((error_name, error_s_v_pairs))
        if visited_pairs is None:
            return error_dict, visited_pairs
        
        # Update the visited pairs list
        assert len(error_s_v_pairs) in [2, 4]

        if ('error_prop' in error_name):
            if 'hall' in error_name:
                visited_pairs.append((error_s_v_pairs[-2], error_s_v_pairs[-1]))
            elif 'miss' in error_name:
                visited_pairs.append((error_s_v_pairs[0], error_s_v_pairs[1]))
        else:
            visited_pairs.append((error_s_v_pairs[0], error_s_v_pairs[1]))
            if len(error_s_v_pairs) == 4:
                visited_pairs.append((error_s_v_pairs[2], error_s_v_pairs[3]))

        return error_dict, visited_pairs

    def preprocess_belief_state(
        self,
        analyzed_item: dict,
        prev_item: dict,
    ):
        # get the previous item's dialogue state and the previous predicted dialogue state
        prev_gold_bs, prev_pred_bs = unroll_or(
            gold=analyzed_item['last_slot_values'], pred=prev_item.get(f'pred_{self.parsing_func.__name__}', {})
        )
        analyzed_item[f'last_pred_{self.parsing_func.__name__}'] = prev_pred_bs

        # Delta(State Change) Belief State parsing. Parse the completion and normalize considering surface forms
        pred_delta_bs = analyzed_item.get(f'pred_delta_{self.parsing_func.__name__}', None)
        if pred_delta_bs is None:
            pred_delta_bs = self.parsing_func(python_completion=analyzed_item['completion'], state=prev_pred_bs)
            pred_delta_bs = self.normalizer.normalize(raw_parse=pred_delta_bs) if 'DELETE' not in str(pred_delta_bs) else pred_delta_bs
            analyzed_item[f'pred_delta_{self.parsing_func.__name__}'] = pred_delta_bs
        gold_delta_bs, pred_delta_bs = unroll_or(gold=analyzed_item['turn_slot_values'], pred=pred_delta_bs)
        
        # Accumulated Dialogue State Belief State. Update the pred with parsed delta(State Change) belief state
        pred_bs = update_dialogue_state(context=prev_pred_bs, normalized_turn_parse=pred_delta_bs)
        gold_bs, pred_bs = unroll_or(gold=analyzed_item['slot_values'], pred=pred_bs)
        analyzed_item[f'pred_{self.parsing_func.__name__}'] = pred_bs

        return gold_bs, pred_bs, gold_delta_bs, pred_delta_bs, prev_gold_bs, prev_pred_bs
    
    
    def detect_delta_missings(
        self, 
        delta_miss_gold: MultiWOZDict,  
        delta_over_pred: MultiWOZDict, 
        gold_delta_bs: MultiWOZDict, 
        pred_delta_bs: MultiWOZDict,
        visited: List[Tuple[SlotName, SlotValue]]
    )-> Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]:
        """
        Detects missing values in the prediction compared to the gold standard and records errors.

        Args:
            delta_miss_gold (MultiWOZDict): Missing values in the gold standard.
            delta_over_pred (MultiWOZDict): Over-predicted values.
            gold_delta_bs (MultiWOZDict): The gold standard delta belief state.
            pred_delta_bs (MultiWOZDict): The predicted delta belief state.
            visited (List[Tuple[SlotName, SlotValue]]): List of visited slot and value pairs.

        Returns:
            Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]: 
            Updated error dictionary and visited pairs list.
        """

        error_dict = defaultdict(list)
        error_dict.setdefault('error', [])
        error_name, error_s_v_pairs = None, None

        for gold_slot, gold_value in delta_miss_gold.items():
            if (gold_slot, gold_value) in visited:
                continue        
            # Confused miss case: the predicted value is the same as the gold value, but the slot is different.
            if (gold_value in delta_over_pred.values()):            
                for (confused_slot, v) in delta_over_pred.items():
                    if v == gold_value and confused_slot != gold_slot:
                        error_name = 'delta_miss_confuse'
                        error_s_v_pairs = (gold_slot, gold_value, confused_slot, v)
                        error_dict, visited = self.record_error_and_update_visited(error_dict, error_name, error_s_v_pairs, visited)
            else:
                # if gold_slot in pred_delta_bs: 'delta_hall_val' error case. But we don't care about it here.
                # if gold_slot not in pred_delta_bs: 'delta_miss_total' error case.
                if gold_slot not in pred_delta_bs:
                    if error_dict.get('error') is None:
                        raise ValueError('Error case is None')
                    error_name = 'delta_miss_total'
                    error_s_v_pairs = (gold_slot, gold_value)
                
                # if gold_value is a special value and not in the prediction, record the error.
                if gold_value in self.special_values and pred_delta_bs.get(gold_slot, None) == None:
                    error_name = f'delta_miss_{re.sub(r"[^a-zA-Z]", "", gold_value)}'
                    error_s_v_pairs = (gold_slot, gold_value)
                    
            error_dict, visited = self.record_error_and_update_visited(error_dict, error_name, error_s_v_pairs, visited) 
        return error_dict, visited

    def detect_delta_hallucinations(
        self,
        delta_miss_gold: MultiWOZDict,
        delta_over_pred: MultiWOZDict,
        gold_delta_bs: MultiWOZDict, 
        pred_delta_bs: MultiWOZDict,
        prev_gold_bs: MultiWOZDict,
        prev_pred_bs: MultiWOZDict,
        visited: List[Tuple[SlotName, SlotValue]]
    )-> Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]:
        """
        Detects hallucinated values in the prediction compared to the gold standard and previous predictions,
        and records errors.

        Args:
            delta_miss_gold (MultiWOZDict): Missing values in the gold standard.
            delta_over_pred (MultiWOZDict): Over-predicted values.
            gold_delta_bs (MultiWOZDict): The gold standard delta belief state.
            pred_delta_bs (MultiWOZDict): The predicted delta belief state.
            prev_pred_bs (MultiWOZDict): The previous predicted belief state.
            visited (List[Tuple[SlotName, SlotValue]]): List of visited slot and value pairs.

        Returns:
            Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]: 
            Updated error dictionary and visited pairs list.
            
        """
        error_dict = defaultdict(list)
        error_dict.setdefault('error', [])
        error_name, error_s_v_pairs, tmp_visited = None, None, []

        for pred_slot, pred_value in delta_over_pred.items():
            if (pred_slot, pred_value) in visited:
                continue
            
            if pred_slot in gold_delta_bs:
                error_name = 'delta_hall_val'
                error_s_v_pairs = (pred_slot, gold_delta_bs[pred_slot], pred_slot, pred_value)
                tmp_visited = visited

            elif pred_slot in prev_pred_bs:
                if pred_value == prev_pred_bs[pred_slot]:
                    continue    # parroting the previous slot, value pair 
                elif prev_gold_bs.get(pred_slot, None) and pred_value == prev_gold_bs[pred_slot]:
                    continue    # or correctly predicting the previous slot, value pair in the current turn
                if pred_value != prev_pred_bs[pred_slot]:
                    error_name = 'delta_hall_overwrite'
                    error_s_v_pairs = (pred_slot, pred_value)
            else:
                error_name = 'delta_hall_total'
                error_s_v_pairs = (pred_slot, pred_value)
            
            error_dict, visited = self.record_error_and_update_visited(error_dict, error_name, error_s_v_pairs, tmp_visited)

        for gold_slot, gold_value in delta_miss_gold.items():
            if (gold_slot, gold_value) in visited:
                continue
            
            if (gold_value not in delta_over_pred.values()):
                if gold_slot in pred_delta_bs:
                    error_name = f'delta_hall_val'
                    error_s_v_pairs = (gold_slot, gold_value, gold_slot, pred_delta_bs[gold_slot])
                
            error_dict, visited = self.record_error_and_update_visited(error_dict, error_name, error_s_v_pairs, visited) 
        
        return error_dict, visited

    def detect_error_propagations(
        self,
        delta_miss_gold, 
        delta_over_pred, 
        gold_bs, 
        pred_bs, 
        prev_gold_bs, 
        prev_pred_bs, 
        visited,
        prev_item: dict,
    )->Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]:
        """
        Analyzes the error cases that are propagated from the previous turn.

        Args:
            delta_miss_gold (MultiWOZDict): Missing values in the gold standard.
            delta_over_pred (MultiWOZDict): Over-predicted values.
            gold_bs (MultiWOZDict): The gold standard belief state.
            pred_bs (MultiWOZDict): The predicted belief state.
            prev_gold_bs (MultiWOZDict): The previous gold standard belief state.
            prev_pred_bs (MultiWOZDict): The previous predicted belief state.
            visited (List[Tuple[SlotName, SlotValue]]): List of visited slot and value pairs.
            prev_item (dict): The previous error log.
        
        Returns:
            Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]: 
            Updated error dictionary and visited pairs list

        """
        prev_over_pred = compute_dict_difference(prev_pred_bs, prev_gold_bs)
        prev_miss_gold = compute_dict_difference(prev_gold_bs, prev_pred_bs)
        
        full_over_pred = compute_dict_difference(pred_bs, gold_bs)
        full_miss_gold = compute_dict_difference(gold_bs, pred_bs)
            
        error_dict = defaultdict(list)
        for err_name, err_s_v in prev_item.get('error', []):
            if 'hall' in err_name:
                error_slot, error_value = err_s_v[-2], err_s_v[-1]
            if 'miss' in err_name :
                error_slot, error_value = err_s_v[0], err_s_v[1]

            if (error_slot, error_value) in visited:
                continue
            
            if (error_slot, error_value) in prev_miss_gold.items() or (error_slot, error_value) in prev_over_pred.items():
                if (error_slot, error_value) in full_over_pred.items() or (error_slot, error_value) in full_miss_gold.items():
                    if 'delete' in err_name:
                        prop_name = 'error_prop_'+'_'.join(err_name.split('_')[-2:])
                        error_dict, visited = self.record_error_and_update_visited(error_dict, prop_name, err_s_v, visited)
                    if (error_slot, error_value) in delta_miss_gold.items() or (error_slot, error_value) in delta_over_pred.items():
                        continue
                    prop_name = 'error_prop_'+'_'.join(err_name.split('_')[-2:])
                    error_dict, visited = self.record_error_and_update_visited(error_dict, prop_name, err_s_v, visited)
        return error_dict, visited
    
    def categorize_error_case(
        self, 
        item: dict, 
        prev_item: dict, 
        gold_bs: MultiWOZDict, 
        pred_bs: MultiWOZDict, 
        gold_delta_bs: MultiWOZDict, 
        pred_delta_bs: MultiWOZDict, 
        prev_gold_bs: MultiWOZDict, 
        prev_pred_bs: MultiWOZDict,
        **kwargs
    ) -> dict:
        """
        Categories the error cases into different types and records them in the log.
        
        Args:
            item (dict): The current error log.
            prev_item (dict): The previous error log.
            gold_bs (MultiWOZDict): The gold standard belief state.
            pred_bs (MultiWOZDict): The predicted belief state.
            gold_delta_bs (MultiWOZDict): The gold standard delta belief state.
            pred_delta_bs (MultiWOZDict): The predicted delta belief state.
            prev_gold_bs (MultiWOZDict): The previous gold standard belief state.
            prev_pred_bs (MultiWOZDict): The previous predicted belief state.
        
        Returns:
            dict: The log updated error cases.
        """
        
        delta_miss_gold = compute_dict_difference(gold_delta_bs, pred_delta_bs)
        delta_over_pred = compute_dict_difference(pred_delta_bs, gold_delta_bs)

        visited = [] if kwargs.get('visited') is None else kwargs.get('visited')

        # handle the case which is already found and recorded in the current turn
        for err_name, err_s_v in item.get('error', []):
            if len(err_s_v) > 2:
                visited.append((err_s_v[-2], err_s_v[-1]))
            visited.append((err_s_v[0], err_s_v[1]))

        # handle the case which prediction missed in the current turn
        error_case, miss_visited = self.detect_delta_missings(
            delta_miss_gold=delta_miss_gold, 
            delta_over_pred=delta_over_pred, 
            gold_delta_bs=gold_delta_bs,
            pred_delta_bs=pred_delta_bs,
            visited=visited
        )
        item['error'].extend(error_case.get('error', []))

        # handle the case which is over-predicted in the current turn
        error_case, hall_visited = self.detect_delta_hallucinations(
            delta_miss_gold=delta_miss_gold,
            delta_over_pred=delta_over_pred,
            gold_delta_bs=gold_delta_bs,
            pred_delta_bs=pred_delta_bs, 
            prev_gold_bs=prev_gold_bs,
            prev_pred_bs=prev_pred_bs, 
            visited=visited
        )
        item['error'].extend(error_case.get('error', []))
        
        # To allow the same (slot, value) pair to be visited multiple times in the miss, hall error cases
        # But, not in the error propagation cases
        visited.extend(miss_visited)
        visited.extend(hall_visited)
        
        # handle the case which is propagated from the previous turn
        error_case, visited = self.detect_error_propagations(
            delta_miss_gold=delta_miss_gold, 
            delta_over_pred=delta_over_pred, 
            gold_bs=gold_bs, 
            pred_bs=pred_bs,
            prev_gold_bs=prev_gold_bs, 
            prev_pred_bs=prev_pred_bs,
            visited=visited, prev_item=prev_item
        )
        item['error'].extend(error_case.get('error', []))

        return item
    
    def reason_error(
        self,
        analyzed_item: dict,
    ):
        """
        대화 맥락을 기반으로 발생한 오류 사례의 원인을 분석합니다.

        Args:
            analyzed_item (dict): 오류를 포함한 대화의 단일 턴 로그.

        Returns:
            dict: 각 오류에 대한 원인이 추가된 업데이트된 로그.
        """
    
        # 대화 맥락, 정답(Gold Standard), 예측 결과를 가져옵니다.
        context = analyzed_item.get('dialog', "")
        gold_bs = analyzed_item.get('slot_values', {})
        pred_bs = analyzed_item.get('pred', {})

        # 분석된 항목의 각 오류를 순회합니다.
        for i, (error_type, error_details) in enumerate(analyzed_item.get('error', [])):
            reason = ""

            # 예시 1: 예측에서 누락된 슬롯 값
            if error_type == "delta_miss_total":
                slot_name, correct_value = error_details
                reason = f"슬롯 '{slot_name}'에 대한 값 '{correct_value}'이(가) 예측에서 누락되었습니다. "
            if slot_name in context:
                reason += f"이 슬롯은 대화 맥락에서 언급되었지만 정확하게 반영되지 않았습니다."

            # 예시 2: 잘못된 값 예측
            elif error_type == "delta_hall_val":
                slot_name, correct_value, pred_value = error_details
                reason = f"슬롯 '{slot_name}'이(가) '{correct_value}'이(가) 아닌 '{pred_value}'로 예측되었습니다. "
            if slot_name in context and correct_value in context:
                reason += f"정확한 값이 맥락에서 언급되었지만 정확하게 인식되지 않았습니다."
            elif pred_value not in context:
                reason += f"예측된 값 '{pred_value}'은(는) 대화 맥락에 나타나지 않습니다."

            # 예시 3: 올바른 값을 잘못된 값으로 덮어쓰기
            elif error_type == "delta_hall_overwrite":
                slot_name, pred_value = error_details
                reason = f"슬롯 '{slot_name}'이(가) 잘못된 값 '{pred_value}'으로 덮어쓰여졌습니다."

            # 예시 4: 이전 턴에서 오류가 전파된 경우
            elif "error_prop" in error_type:
                slot_name, correct_value = error_details[:2]
                reason = f"이 오류는 이전 턴에서 전파되었습니다. 슬롯 '{slot_name}'이(가) '{correct_value}' 값을 가지고 있어 오류를 발생시켰습니다."

            # 오류에 대한 원인을 analyzed_item에 추가합니다.
            analyzed_item['error'][i] = (error_type, error_details, reason)

        return analyzed_item
    
    def analyze(self):
        """
        Analyzes the errors in the prediction compared to the gold standard and records them.
        """
        print('Start analyzing the error cases...')
        logs = read_json(self.result_file_path)
        
        analyzed_log = []
        n_correct = 0
        prev_item = {}
        for idx, data_item in tqdm(enumerate(logs), desc="analyzing items", total=len(logs)):
            if data_item['turn_id'] == 0:
                prev_item = {}            

            analyzed_item = copy.deepcopy(data_item)
            
            (   
                gold_bs, pred_bs, 
                gold_delta_bs, pred_delta_bs, 
                prev_gold_bs, prev_pred_bs
            ) = self.preprocess_belief_state(analyzed_item, prev_item)
            
            if pred_bs==gold_bs:
                n_correct+=1

            analyzed_item['error'] = []

            analyzed_item = self.categorize_error_case(
                item=analyzed_item, prev_item=prev_item, 
                gold_bs=gold_bs, 
                pred_bs=pred_bs, 
                gold_delta_bs=gold_delta_bs, 
                pred_delta_bs=pred_delta_bs, 
                prev_gold_bs=prev_gold_bs, 
                prev_pred_bs=prev_pred_bs
            )
            
            # remove the redundant error cases
            analyzed_item['error'] = sorted(list(set(tuple(x) for x in analyzed_item['error'])))

            # analyzed_item = self.reason_error(analyzed_item)
            
            analyzed_item = sort_data_item(data_item=analyzed_item, parsing_func=self.parsing_func.__name__)
            analyzed_log.append(analyzed_item)
            prev_item = analyzed_item

            if idx % 1000 == 0:
                save_analyzed_log(output_dir_path=self.output_dir_path, analyzed_log=analyzed_log)
        
        save_analyzed_log(output_dir_path=self.output_dir_path, analyzed_log=analyzed_log)
        return analyzed_log

    def show_stats(self, analyzed_log):
        total_acc, total_f1 = 0, 0
        jga_by_turn_id = defaultdict(list)  # use to record the accuracy
        jga_by_dialog = defaultdict(list)  # use to record the accuracy
        wrong_smaples = []
        n_correct = 0
        n_total = len(analyzed_log)
        for data_item in analyzed_log:
            pred = data_item['pred']
            this_jga, this_acc, this_f1 = evaluate(pred, data_item['slot_values'])
            total_acc += this_acc
            total_f1 += this_f1
            if this_jga:
                n_correct += 1
                jga_by_turn_id[data_item['turn_id']].append(1)
                jga_by_dialog[data_item['ID']].append(1)
            else:
                jga_by_turn_id[data_item['turn_id']].append(0)
                jga_by_dialog[data_item['ID']].append(0)
                wrong_smaples.append(data_item)

        stats = evaluate_logs(analyzed_log, test_set=analyzed_log)
        slot_prf = slot_level_f1(analyzed_log, tp_means_correct=True)

        slot_acc: Dict[str, Counter] = defaultdict(Counter)
        for turn in tqdm(analyzed_log, desc="calculating slot-level F1", total=len(analyzed_log)):
            for gold_slot, gold_value in turn['slot_values'].items():
                slot_acc[gold_slot]['total'] += 1
                if gold_slot in turn['pred'] and turn['pred'][gold_slot] == gold_value:
                    slot_acc[gold_slot]['right'] += 1

        slot_acc = {slot: slot_acc[slot]['right'] / slot_acc[slot]['total'] for slot in slot_acc}
        slot_acc = dict(sorted(slot_acc.items(), key=lambda x: x[0]))
        
        slot_f1 = {k: v[1] for k, v in slot_prf.items()}
        slot_f1 = dict(sorted(slot_f1.items(), key=lambda x: x[0]))

        stats_df = pd.DataFrame(slot_acc.items(), columns=['slot', 'acc'])
        stats_df.merge(pd.DataFrame(slot_f1.items(), columns=['slot', 'f1']), on='slot')

        print(stats_df)
        stats_df.to_csv(self.output_dir_path + '/slot_acc_f1.csv', index=False)
        
        return None

    def make_confusion_matrix(self, analyzed_log, criteria='value'):
        """
        Predicted State Change
        """
        slots = list(self.normalizer.ontology.valid_slots)
        pred_delta_confusion_matrix = {
            gold_slot: {pred_slot:0 for pred_slot in slots} 
            for gold_slot in slots+['hall_value', 'hall_total']
        }
        pred_delta_confusion_matrix['text_hallucination'] = 0

        gold_delta_confusion_matrix = {
            gold_slot: {pred_slot:0 for pred_slot in slots+['miss_value', 'miss_total']} 
            for gold_slot in slots
        }

        for data_item in analyzed_log:
            
            pred_delta_bs = data_item[f'pred_delta_{self.parsing_func.__name__}']
            gold_delta_bs = data_item['turn_slot_values']

            # filter out text hallucination totally wrong generation
            if pred_delta_bs == {} and 'update' not in data_item['completion']: 
                for slot_name in gold_delta_bs:
                    pred_delta_confusion_matrix['text_hallucination'] += 1
                    continue
            
            if criteria == 'value':
                pred_delta_confusion_matrix, gold_delta_confusion_matrix = self._update_conf_mat_value(
                    pred_delta_bs, gold_delta_bs, 
                    pred_delta_confusion_matrix, gold_delta_confusion_matrix
                )
            elif criteria == 'slot':
                pred_delta_confusion_matrix, gold_delta_confusion_matrix = self._update_conf_mat_slot(
                    pred_delta_bs, gold_delta_bs, 
                    pred_delta_confusion_matrix, gold_delta_confusion_matrix
                )
            else:
                raise ValueError('Invalid criteria')
                    
        return pred_delta_confusion_matrix, gold_delta_confusion_matrix

    def _update_conf_mat_value(
        self, 
        pred_delta_bs, gold_delta_bs,
        pred_delta_confusion_matrix, 
        gold_delta_confusion_matrix
    ):
        for pred_slot, pred_value in pred_delta_bs.items():
            if pred_value in list(gold_delta_bs.values()):
                for gold_slot in [k for k, v in gold_delta_bs.items() if v == pred_value]:
                    pred_delta_confusion_matrix[gold_slot][pred_slot] += 1
            else:
                if pred_slot in list(gold_delta_bs.keys()):
                    # value hall: pred: {gold_slot: wrong_value}
                    pred_delta_confusion_matrix['hall_value'][pred_slot] += 1
                else:
                    # hall_total: pred: {wrong_slot: wrong_value}
                    pred_delta_confusion_matrix['hall_total'][pred_slot] += 1

        for gold_slot, gold_value in gold_delta_bs.items():
            if gold_value in list(pred_delta_bs.values()):
                for pred_slot in [k for k, v in pred_delta_bs.items() if v == gold_value]:
                    gold_delta_confusion_matrix[gold_slot][pred_slot] += 1
            else:
                if gold_slot in list(pred_delta_bs.keys()):
                    # value miss: pred:{gold_slot: wrong_value}
                    gold_delta_confusion_matrix[gold_slot]['miss_value'] += 1
                else:
                    # miss_total: pred: {wrong_slot: wrong_value}
                    gold_delta_confusion_matrix[gold_slot]['miss_total'] += 1
        
        return pred_delta_confusion_matrix, gold_delta_confusion_matrix
    
    def _update_conf_mat_slot(
        self,
        pred_delta_bs, gold_delta_bs,
        pred_delta_confusion_matrix,
        gold_delta_confusion_matrix
    ):
        for pred_slot, pred_value in pred_delta_bs.items():
            if pred_slot not in list(gold_delta_bs.keys()):
                if pred_value in list(gold_delta_bs.values()):
                    for gold_slot in [k for k, v in gold_delta_bs.items() if v == pred_value and k != pred_slot]:
                        pred_delta_confusion_matrix[gold_slot][pred_slot] += 1
                else:
                    pred_delta_confusion_matrix['hall_total'][pred_slot] += 1
            else:
                if pred_value == gold_delta_bs[pred_slot]:
                    pred_delta_confusion_matrix[pred_slot][pred_slot] += 1
                else:
                    pred_delta_confusion_matrix['hall_value'][pred_slot] += 1
                
        for gold_slot, gold_value in gold_delta_bs.items():
            if gold_slot in list(pred_delta_bs.keys()):
                if gold_value == pred_delta_bs[gold_slot]:
                    gold_delta_confusion_matrix[gold_slot][gold_slot] += 1
                else:
                    gold_delta_confusion_matrix[gold_slot]['miss_value'] += 1
            else:
                if gold_value in list(pred_delta_bs.values()):
                    for pred_slot in [k for k, v in pred_delta_bs.items() if v == gold_value and k != gold_slot]:
                        gold_delta_confusion_matrix[gold_slot][pred_slot] += 1
                else:
                    gold_delta_confusion_matrix[gold_slot]['miss_total'] += 1

        return pred_delta_confusion_matrix, gold_delta_confusion_matrix

    def show_state_change_confusion_matrix(self):
        analyzed_log = load_analyzed_log(self.output_dir_path)
        pred_delta_conf_mat, gold_delta_conf_mat = self.make_confusion_matrix(analyzed_log, criteria='value')

        slots = list(self.normalizer.ontology.valid_slots)
        # Convert the confusion matrix to the pandas DataFrame
        fig, axes = plt.subplots(1,2, figsize=(30, 15))
        pred_delta_df = pd.DataFrame(pred_delta_conf_mat).T.drop('text_hallucination', axis=0).dropna()
        gold_delta_df = pd.DataFrame(gold_delta_conf_mat).T.dropna()

        sns.heatmap(pred_delta_df, mask=(pred_delta_df == 0), annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='black', ax=axes[0])
        sns.heatmap(gold_delta_df, mask=(gold_delta_df == 0), annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='black', ax=axes[1])

        axes[0].set_xlabel('Predicted State Change')
        axes[0].set_ylabel('Gold State Change')
        axes[1].set_xlabel('Predicted State Change')
        axes[1].set_ylabel('Gold State Change')

        print('Confusion matrix saved to the output directory')
        print('PRED Confusion matrix')
        print("# of hal value (Right SLOT Wrong VALUE): ", pred_delta_df.loc['hall_value'].sum())
        print("# of confused (Wrong SLOT Right VALUE): ", pred_delta_df.loc[slots].values.sum() - pred_delta_df.loc[slots].values.diagonal().sum())
        print("# of hall total (Wrong SLOT Wrong VALUE): ", pred_delta_df.loc['hall_total'].sum())
        print()
        print('GOLD Confusion matrix')
        print("# of miss value (Right SLOT Wrong VALUE): ", gold_delta_df.loc[:, 'miss_value'].sum())
        print("# of confused (Wrong SLOT Right VALUE): ", gold_delta_df.loc[slots].values.sum() - gold_delta_df.loc[slots].values.diagonal().sum())
        print("# of miss total (Wrong SLOT Wrong VALUE): ", gold_delta_df.loc[:, 'miss_total'].sum())
        
        plt.savefig(self.output_dir_path + '/confusion_matrix.png')
        return None
    

if __name__ == '__main__':
    analyzer = ErrorAnalyzer(
        train_data_path='/home/haesungpyun/my_refpydst/data/mw21_0p_train.json',
        result_file_path='/home/haesungpyun/my_refpydst/outputs/runs/table4_llama/zero_shot/split_v1_greedy_0620_1337/running_log.json',
        output_dir_path='./outputs/',
    )
    analyzer.analyze()

    analyzer.show_state_change_confusion_matrix()