import os
import json
from loaders import graph_folder_path, gold_folder_path, load_gold
import torch
from sentence_transformers import util, SentenceTransformer


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


embeddings_path = "/home/vassm/entity_alignment/kg_entity_alignment_2024/features/embeddings/version2"
embeddings_postfix = "_lab_altlab_type_abs_comment_BAAI_bge-large-en-v1.5.json"

topk = 10
output_path = f"/home/vassm/entity_alignment/kg_entity_alignment_2024/features/top{topk}pairs/bgelarge_v1"

golds = list(filter(lambda x: ".xml" in x, os.listdir(os.path.split(gold_folder_path)[0])))
if not os.path.exists(output_path):
    os.makedirs(output_path)

output_container = dict()
print("Output path:", output_path)
print("TOPK:", topk)

for gold in golds:
    print(gold)
    loaded_gold = load_gold(gold)
    loaded_gold = loaded_gold[0]

    g1, g2 = gold.replace(".xml", "").split("-")
    if g1 == "marvelcinematicuniverse":
        g1 = "mcu"
    with open(os.path.join(embeddings_path, g1 + embeddings_postfix), "r") as f:
        g1_embedding = json.load(f)
    with open(os.path.join(embeddings_path, g2 + embeddings_postfix), "r") as f:
        g2_embedding = json.load(f)

    graph1_path = os.path.join(graph_folder_path, g1 + ".triples")
    with open(graph1_path.replace(".triples", "_mapping.json"), "r") as f:
        g1_name2id = json.load(f)
        g1_id2name = dict((v, k) for k, v in g1_name2id.items())

    graph2_path = os.path.join(graph_folder_path, g2 + ".triples")
    with open(graph2_path.replace(".triples", "_mapping.json"), "r") as f:
        g2_name2id = json.load(f)
        g2_id2name = dict((v, k) for k, v in g2_name2id.items())

    g1_torch_embeds = torch.Tensor(list(g1_embedding.values()))
    g2_torch_embeds = torch.Tensor(list(g2_embedding.values()))
    pair_top10 = util.semantic_search(g1_torch_embeds, g2_torch_embeds, top_k=topk)
    reverse_pair_top10 = util.semantic_search(g2_torch_embeds, g1_torch_embeds, top_k=topk)

    forward_dict = dict()
    backward_dict = dict()
    g2_keys = list(g2_embedding.keys())
    g1_keys = list(g1_embedding.keys())
    for a, b in zip(list(g1_embedding.keys()), pair_top10):
        row_info = list()
        for element in b:
            row_info.append([g2_keys[element["corpus_id"]], element["score"]])
        forward_dict[str(a)] = row_info

    for a, b in zip(list(g2_embedding.keys()), reverse_pair_top10):
        row_info = list()
        for element in b:
            row_info.append([g1_keys[element["corpus_id"]], element["score"]])
        backward_dict[str(a)] = row_info

    with open(os.path.join(output_path, f"{g1}-{g2}_top{str(topk)}pairs.json"), "w") as f:
        json.dump(forward_dict, f)

    with open(os.path.join(output_path, f"{g2}-{g1}_top{str(topk)}pairs.json"), "w") as f:
        json.dump(backward_dict, f)
