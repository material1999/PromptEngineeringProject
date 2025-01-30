from openai import OpenAI

import os
import json
from loaders import graph_folder_path, gold_folder_path, load_gold
from tqdm import tqdm


def query_gpt_api(client, text):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        # model="gpt-4o",
        messages=[
            {"role": "system", "content": text}
        ],
        max_tokens=4096,
    )
    return completion.choices[0].message.content


def construct_prompt(anchor, candidates):
    candidate_str = ""
    for i, candidate in enumerate(candidates):
        candidate_str += "<EXAMPLE>\nID:{}\n".format(str(i + 1))
        candidate_str += f"{candidate}\n"
        candidate_str += "<\\EXAMPLE>\n"

    return f"""TASK: You will be given a description of an anchor entity and a list of candidate entities, all formatted with XML tags. Your task is to:
- Identify the candidate entity that matches the anchor entity.
- Return the ID number of the matching candidate entity.
- If none of the candidates match the anchor entity, return -1.
###
Anchor: {anchor}
###
{candidate_str}
###
Answer:"""


def extract_missing_golds(gold, pairs):
    remaining = list()

    for g in gold:
        flag = False
        for p in pairs:
            if g[0] == p[0] and g[1] == p[1]:
                flag = True
                break
        if flag is False:
            remaining.append(g)
    return remaining


def class_filter(elements):
    remaining = list()
    for elem in elements:
        if "/class/" in elem[0] or "/class/" in elem[1]:
            continue
        else:
            remaining.append(elem)
    return remaining


def property_filter(elements):
    remaining = list()
    for elem in elements:
        if "/property/" in elem[0] or "/property/" in elem[1]:
            continue
        else:
            remaining.append(elem)
    return remaining


client = OpenAI(
    api_key="<API_KEY>")

exactmatch_path = "/home/vassm/entity_alignment/kg_entity_alignment_2024/outputs/exact_match/label_altlabel"
dogtags_path = "/home/vassm/entity_alignment/kg_entity_alignment_2024/features/dogtags/version2"
top10pairs_path = "/home/vassm/entity_alignment/kg_entity_alignment_2024/features/top10pairs/bgelarge_v1"

output_path = "/home/vassm/entity_alignment/kg_entity_alignment_2024/outputs/gpt4omini/top10pair_selector_v1"

golds = list(filter(lambda x: ".xml" in x, os.listdir(os.path.split(gold_folder_path)[0])))
if not os.path.exists(output_path):
    os.makedirs(output_path)

for gold in golds[:4]:
    print(gold)
    loaded_gold = load_gold(gold)
    loaded_gold = loaded_gold[0]

    with open(os.path.join(exactmatch_path, gold.replace(".xml", ".json")), "r") as f:
        exactmatches = json.load(f)

    g1, g2 = gold.replace(".xml", "").split("-")
    if g1 == "marvelcinematicuniverse":
        g1 = "mcu"

    graph1_path = os.path.join(graph_folder_path, g1 + ".triples")
    with open(graph1_path.replace(".triples", "_mapping.json"), "r") as f:
        g1_name2id = json.load(f)
        g1_id2name = dict((v, k) for k, v in g1_name2id.items())

    graph2_path = os.path.join(graph_folder_path, g2 + ".triples")
    with open(graph2_path.replace(".triples", "_mapping.json"), "r") as f:
        g2_name2id = json.load(f)
        g2_id2name = dict((v, k) for k, v in g2_name2id.items())

    with open(os.path.join(dogtags_path, g1 + "_lab_altlab_type_abs_comment_deduplicated.json"), "r") as f:
        g1_dogtags = json.load(f)

    with open(os.path.join(dogtags_path, g2 + "_lab_altlab_type_abs_comment_deduplicated.json"), "r") as f:
        g2_dogtags = json.load(f)

    with open(os.path.join(top10pairs_path, g1 + "-" + g2 + "_top10pairs.json"), "r") as f:
        forward_pairs = json.load(f)

    with open(os.path.join(top10pairs_path, g2 + "-" + g1 + "_top10pairs.json"), "r") as f:
        backward_pairs = json.load(f)

    missing_gold = extract_missing_golds(loaded_gold, exactmatches)

    missing_gold_filtered = class_filter(missing_gold)
    missing_gold_filtered = property_filter(missing_gold_filtered)

    top10_selections_forward = list()
    top10_selections_backward = list()

    top10_selections_unioned = list()

    errors = list()
    for gold_pair in tqdm(missing_gold_filtered):
        g1_id = g1_name2id[gold_pair[0]]
        g2_id = g2_name2id[gold_pair[1]]
        gold1_dogtag = g1_dogtags[str(g1_id)]
        gold2_dogtag = g2_dogtags[str(g2_id)]
        candidate_g2_dogtags = [g2_dogtags[str(element[0])] for element in forward_pairs[str(g1_id)]]
        candidate_g1_dogtags = [g1_dogtags[str(element[0])] for element in backward_pairs[str(g2_id)]]

        prompt = construct_prompt(gold1_dogtag, candidate_g2_dogtags)
        response_f = query_gpt_api(client, prompt)
        response_forward = None
        response_backward = None
        try:
            response_forward = int(response_f.replace("ID:", ""))
        except Exception as e:
            errors.append([prompt, response_f, gold_pair, g1, g2])
            print("Forward:", e)

        if response_forward != -1 and response_forward is not None:
            top10_selections_forward.append([g1_id, forward_pairs[str(g1_id)][response_forward - 1]])

        prompt = construct_prompt(gold2_dogtag, candidate_g1_dogtags)
        response_b = query_gpt_api(client, prompt)
        try:
            response_backward = int(response_b.replace("ID:", ""))
        except Exception as e:
            errors.append([prompt, response_b, gold_pair, g2, g1])
            print("Backward:", e)

        if response_backward != -1 and response_backward is not None:
            top10_selections_backward.append([g2_id, backward_pairs[str(g2_id)][response_backward - 1]])

        if (response_backward != -1 and
                response_backward is not None and
                response_forward != -1 and
                response_forward is not None):
            if (backward_pairs[str(g2_id)][response_backward - 1][0] == str(g1_id) and
                    forward_pairs[str(g1_id)][response_forward - 1][0] == str(g2_id)):
                top10_selections_unioned.append([gold_pair[0], gold_pair[1]])

    print(errors)
    print("ERRORS:", len(errors))
    readable_top10_selections_forward = [[g1_id2name[element[0]],
                                          [g2_id2name[int(element[1][0])],
                                           element[1][1]]] for element in top10_selections_forward]
    readable_top10_selections_backward = [[g2_id2name[element[0]],
                                           [g1_id2name[int(element[1][0])],
                                            element[1][1]]] for element in top10_selections_backward]

    with open(os.path.join(output_path, g1 + "-" + g2 + "_top10pair_selected_forward.json"), "w") as f:
        json.dump(readable_top10_selections_forward, f)
    with open(os.path.join(output_path, g1 + "-" + g2 + "_top10pair_selected_backward.json"), "w") as f:
        json.dump(readable_top10_selections_backward, f)
    with open(os.path.join(output_path, g1 + "-" + g2 + "_top10pair_selected_unioned.json"), "w") as f:
        json.dump(top10_selections_unioned, f)
