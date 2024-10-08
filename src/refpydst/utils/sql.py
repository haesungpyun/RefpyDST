"""
This file was adapted from the code for the paper "In Context Learning for Dialogue State Tracking", as originally
published here: https://github.com/Yushi-Hu/IC-DST. Cite their article as:

@article{hu2022context,
  title={In-Context Learning for Few-Shot Dialogue State Tracking},
  author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
  journal={arXiv preprint arXiv:2203.08568},
  year={2022}
}
"""
import re
from collections import OrderedDict
from typing import Dict

import sqlparse
from refpydst.data_types import MultiWOZDict


def slot_values_to_seq_sql(original_slot_values: MultiWOZDict, single_answer: bool = False) -> str:
    """
    Given a set of slot value pairs in MultiWOZ normalized form, return a SQL SELECT statement which encodes
    the corresponding search. In prompting in the IC-DST experiments, the original_slot_values argument would be a turn
    level delta

    :param original_slot_values: slot values to encode
    :param single_answer: true if the slot values only have one answer vs. a | delimited string
    :return: SQL statement representing the slot values
    """
    sql_str = ""
    tables = OrderedDict()
    col_value = dict()

    # add '_' in SQL columns
    slot_values = {}
    for slot, value in original_slot_values.items():
        if ' ' in slot:
            slot = slot.replace(' ', '_')
        slot_values[slot] = value

    for slot, value in slot_values.items():
        assert len(slot.split("-")) == 2

        if '|' in value:
            value = value.split('|')[0]

        table, col = slot.split("-")  # slot -> table-col

        if table not in tables.keys():
            tables[table] = []
        tables[table].append(col)

        # sometimes the answer is ambiguous
        if single_answer:
            value = value.split('|')[0]
        col_value[slot] = value

    # When there is only one table
    if len(tables.keys()) == 1:
        where_clause = []
        table = list(tables.keys())[0]
        for col in tables[table]:
            where_clause.append("{} = {}".format(col, col_value["{}-{}".format(table, col)]))
        sql_str = "SELECT * FROM {} WHERE {}".format(table, " AND ".join(where_clause))
    # When there are more than one table
    else:
        # We observed that Codex has variety in the table short names, here we just use a simple version.
        from_clause = []
        where_clause = []
        for i, table in enumerate(tables.keys()):
            t_i = "t{}".format(i + 1)
            from_clause.append("{} AS {}".format(table, t_i))
            for col in tables[table]:
                where_clause.append("{}.{} = {}".format(t_i, col, col_value["{}-{}".format(table, col)]))
        sql_str = "SELECT * FROM {} WHERE {}".format(", ".join(from_clause), " AND ".join(where_clause))

    return sql_str


def sql_pred_parse(pred_completion) -> Dict[str, str]:
    # parse sql results and fix general errors
    try:
        pred_completion = " * FROM" + pred_completion

        # fix for no states
        if pred_completion == " * FROM  WHERE ":
            return {}

        # remove limit statements
        pred_completion = re.sub(r" LIMIT [0-9]*", "", pred_completion)

        # Here we need to write a parser to convert back to dialogue state
        pred_slot_values = []
        # pred = pred.lower()
        parsed = sqlparse.parse(pred_completion)
        if not parsed:
            return {}
        stmt = parsed[0]
        sql_toks = pred_completion.split()
        operators = [" = ", " LIKE ", " < ", " > ", " >= ", " <= "]

        if "AS" in pred_completion:
            as_indices = [i for i, x in enumerate(sql_toks) if x == "AS"]

            table_name_map_dict = {}
            for indice in as_indices:
                # hotel AS h -> {h: hotel}
                table_name_map_dict[sql_toks[indice + 1].replace(",", "")] = sql_toks[indice - 1]

            slot_values_str = str(stmt.tokens[-1]).replace("_", " ").replace("""'""", "").replace("WHERE ", "")
            for operator in operators:
                slot_values_str = slot_values_str.replace(operator, "-")
            slot_values = slot_values_str.split(" AND ")

            for sv in slot_values:
                for t_ in table_name_map_dict.keys():
                    sv = sv.replace(t_ + ".", table_name_map_dict[t_] + "-")
                pred_slot_values.append(sv)
        else:

            table_name = sql_toks[sql_toks.index("FROM") + 1]

            slot_values_str = str(stmt.tokens[-1]).replace("_", " ").replace("""'""", "").replace("WHERE ", "")
            for operator in operators:
                slot_values_str = slot_values_str.replace(operator, "-")
            slot_values = slot_values_str.split(" AND ")

            pred_slot_values.extend([table_name + "-" + sv for sv in slot_values if slot_values != ['']])

        pred_slot_values = {'-'.join(sv_pair.split('-')[:-1]): sv_pair.split('-')[-1] for sv_pair in pred_slot_values}

        # remove _ in SQL columns
        pred_slot_values = {slot.replace('_', ' '): value for slot, value in pred_slot_values.items()}
        return pred_slot_values
    except Exception as e:
        return {}