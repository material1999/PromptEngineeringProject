from fuzzywuzzy import fuzz
from collections import Counter


def select_top1_left(pairs, element):
    best_score = 0
    best_element = None
    for pair in pairs:
        if pair[0] == element:
            score = fuzz.ratio(pair[0], pair[1])
            if score > best_score:
                best_score = score
                best_element = pair
    return best_element


def select_top1_right(pairs, element):
    best_score = 0
    best_element = None
    for pair in pairs:
        if pair[1] == element:
            score = fuzz.ratio(pair[0], pair[1])
            if score > best_score:
                best_score = score
                best_element = pair
    return best_element


def deduplicate_leftright(pairs):
    elem0 = Counter([element[0] for element in pairs])

    remaining = list()
    for k, v in elem0.items():
        if v > 1:
            remaining.append(select_top1_left(pairs, k))
        else:
            for pair in pairs:
                if k == pair[0]:
                    remaining.append(pair)
                    break

    elem1 = Counter([element[1] for element in remaining])
    remaining2 = list()
    for k, v in elem1.items():
        if v >= 1:
            remaining2.append(select_top1_right(remaining, k))
        else:
            for pair in pairs:
                if k == pair[1]:
                    remaining2.append(pair)
                    break

    return remaining2


def deduplicate_rightleft(pairs):
    elem1 = Counter([element[1] for element in pairs])

    remaining = list()
    for k, v in elem1.items():
        if v > 1:
            remaining.append(select_top1_right(pairs, k))
        else:
            for pair in pairs:
                if k == pair[1]:
                    remaining.append(pair)
                    break
    elem0 = Counter([element[0] for element in remaining])
    remaining2 = list()
    for k, v in elem0.items():
        if v >= 1:
            remaining2.append(select_top1_left(remaining, k))
        else:
            for pair in pairs:
                if k == pair[0]:
                    remaining2.append(pair)
                    break

    return remaining2


import os
import json
from loaders import gold_folder_path, load_gold

exactmatch_path = "/home/vassm/entity_alignment/kg_entity_alignment_2024/outputs/exact_match/label_altlabel"
output_path = "/home/vassm/entity_alignment/kg_entity_alignment_2024/outputs/exact_match_deduplicated"

golds = list(filter(lambda x: ".xml"in x, os.listdir(os.path.split(gold_folder_path)[0])))


for gold in golds:
    print(gold)
    loaded_gold = load_gold(gold)

    with open(os.path.join(exactmatch_path, gold.replace(".xml", ".json")), "r") as f:
        exactmatches = json.load(f)

    leftright_dedup = deduplicate_leftright(exactmatches)
    rightleft_dedup = deduplicate_rightleft(exactmatches)

    with open(os.path.join(output_path, gold.replace(".xml", "_leftright.json")), "w") as f:
        json.dump(leftright_dedup, f)
    with open(os.path.join(output_path, gold.replace(".xml", "_rightleft.json")), "w") as f:
        json.dump(rightleft_dedup, f)
