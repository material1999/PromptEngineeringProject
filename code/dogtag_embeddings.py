from sentence_transformers import SentenceTransformer
import os
import json
from loaders import load_graph, graph_folder_path
from tqdm import tqdm
import re

selected_edge_types = {
    "http://www.w3.org/2000/01/rdf-schema#label": "Label",
    "http://www.w3.org/2004/02/skos/core#altLabel": "Alternative Label",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type": "Type",
    "http://dbkwik.webdatacommons.org/ontology/abstract": "Abstract",
    "http://www.w3.org/2000/01/rdf-schema#comment": "Comment",
}

dogtags_output_path = "/home/vassm/entity_alignment/kg_entity_alignment_2024/features/dogtags/version2"
embeddings_output_path = "/home/vassm/entity_alignment/kg_entity_alignment_2024/features/embeddings/version2"


def convert_triple_to_names(triple, id2name):
    return [id2name[triple[0]], id2name[triple[1]], id2name[triple[2]["edge_label"]]]


def convert_dogtag_container_to_dogtag_string(container, link_list):
    dogtags = dict()

    for k, v in container.items():
        element_strings = []
        for k2, v2 in link_list.items():
            if v2 not in v:
                continue
            if v2 == "Comment" and "Abstract" in v and v[v2] == v["Abstract"]:
                # print("Skipping Comment:", v[v2], "==", v["Abstract"])
                continue
            if v2 == "Alternative Label" and "Label" in v and v[v2] == v["Label"]:
                continue
            if v2 == "Type":
                # element_str += "/".join(v[v2].split("/")[-2:])
                element_str = v[v2].split("/")[-1]
            else:
                element_str = v[v2]
            element_strings.append(element_str)
        element_string = "\n".join(element_strings)
        dogtags[k] = element_string
    return dogtags


dogtag_exclusions = [".jpg", ".jpeg", ".png", ".gif"]


def construct_dogtags(g, name2id, id2name, link_list):
    dogtag_container = dict()

    for edge in g.edges(data=True):
        skip = False
        for exclusion in dogtag_exclusions:
            if exclusion in id2name[edge[0]]:
                skip = True
        if skip:
            continue
        string_edge = id2name[edge[2]["edge_label"]]
        if string_edge in link_list:
            string_triple = convert_triple_to_names(edge, id2name)
            if edge[0] not in dogtag_container:
                dogtag_container[edge[0]] = {
                    link_list[string_edge]: re.sub(r'\n+', '\n', string_triple[1]).strip()
                }
            else:
                dogtag_container[edge[0]][link_list[string_edge]] = re.sub(r'\n+', '\n', string_triple[1]).strip()
    dogtag_strings = convert_dogtag_container_to_dogtag_string(dogtag_container, link_list)
    return dogtag_strings


def sentence_embed(dogtags, model_name="dunzhang/stella_en_400M_v5", gpu=False):
    if gpu:
        model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda").cuda()
    else:
        model = SentenceTransformer(model_name, trust_remote_code=True, device="cpu")

    embeddings = dict()

    for key, dt in tqdm(dogtags.items()):
        query_embedding = model.encode(dt,
                                       # prompt_name=query_prompt_name
                                       ).tolist()
        embeddings[key] = query_embedding
    return embeddings


def write_dogtags(graph_name, dogtags, extra_identifier=""):
    filename = graph_name.replace(".xml", "")
    if extra_identifier is not None:
        filename += f"_{extra_identifier}"
    filename = filename.replace("/", "_")
    filename += ".json"

    with open(os.path.join(dogtags_output_path, filename), "w") as f:
        json.dump(dogtags, f)


def write_embeddings(graph_name, embeddings, extra_identifier=""):
    filename = graph_name.replace(".xml", "")
    if extra_identifier is not None:
        filename += f"_{extra_identifier}"
    filename = filename.replace("/", "_")
    filename += ".json"

    with open(os.path.join(embeddings_output_path, filename), "w") as f:
        json.dump(embeddings, f)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu = True

graphs = list(filter(lambda x: ".xml" in x, os.listdir(os.path.split(graph_folder_path)[0])))

bge_large = "BAAI/bge-large-en-v1.5"
minilm = "all-MiniLM-L6-v2"


selected_embedder = bge_large
print("Model:", selected_embedder)
for g in graphs[::-1]:
    print(g)
    nx_graph, name2id = load_graph(os.path.join(graph_folder_path, g.replace(".xml", ".triples")))
    id2name = dict((v, k) for k, v in name2id.items())
    dogtags = construct_dogtags(nx_graph, name2id, id2name, selected_edge_types)
    print("Writing DogTags...")
    write_dogtags(g, dogtags, "lab_altlab_type_abs_comment_deduplicated")
    print("Done...")
    embeddings = sentence_embed(dogtags, model_name=selected_embedder, gpu=gpu)
    print("Writing Embeddings...")
    write_embeddings(g, embeddings, f"lab_altlab_type_abs_comment_{selected_embedder}")
    print("Done...")
