def get_left_recall(loaded_gold, gold1, g1_name2id, g2_name2id, g2_embedding, forward_top):
    fns = list()
    tps = list()

    for goldpair in loaded_gold:
        ith = list(gold1.keys()).index(str(g1_name2id[goldpair[0]]))
        ith2 = list(g2_embedding.keys()).index(str(g2_name2id[goldpair[1]]))

        found = False
        for element in forward_top[ith]:
            if str(element['corpus_id']) == str(ith2):
                tps.append(goldpair)
                found = True
                break
        if not found:
            fns.append(goldpair)
    recall = len(tps)/(len(tps) + len(fns))
    return recall, len(tps), len(fns)


def get_right_recall(loaded_gold, gold2, g1_name2id, g2_name2id, g1_embedding, backward_top):
    fns = list()
    tps = list()

    for goldpair in loaded_gold:
        ith = list(gold2.keys()).index(str(g2_name2id[goldpair[1]]))
        ith2 = list(g1_embedding.keys()).index(str(g1_name2id[goldpair[0]]))

        found = False
        for element in backward_top[ith]:
            if str(element['corpus_id']) == str(ith2):
                tps.append(goldpair)
                found = True
                break
        if not found:
            fns.append(goldpair)
    recall = len(tps)/(len(tps) + len(fns))
    return recall, len(tps), len(fns)
