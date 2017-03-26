import data_util
import train_iterator


def read_dev(kbest_filename, gold_filename, vocab):
    with open(kbest_filename, 'r') as reader:
        kbest_data = reader.readlines()
    kbest_data.append('PTB_KBEST')
    reader.close()
    kbest = []
    scores = []
    onebest = []
    tree = []
    onescores = []
    lines = []
    onelines = []
    i = 0
    while i < len(kbest_data):
        line = kbest_data[i]
        if line.strip() != 'PTB_KBEST':
            if line.strip() == '':
                onelines.append(tree[:])
                onebest.append(train_iterator.read_tree(tree,vocab))
                tree = []
            elif not '_' in line:
                onescores.append(float(line))
            else:
                tree.append(line)
        else:
            if len(onebest) > 1:
                kbest.append(onebest[:])
                scores.append(onescores[:])
                lines.append(onelines)
                onelines = []
                onebest = []
                onescores = []
        i += 1

    with open(gold_filename, 'r') as reader:
        data = reader.readlines()
    reader.close()
    list = []
    gold = []
    gold_lines = []
    for line in data:
        if line.strip() == '':
            root = train_iterator.read_tree(list,vocab)
            gold.append(root)
            gold_lines.append(list[:])
            list = []
        else:
            list.append(line)
    dev_data = []
    for a,b,c,d,e in zip(kbest, scores, gold, lines, gold_lines):
        if len(c.children) == 0:
            continue
        dev_data.append(data_util.instance(a,b,c,d,e))
    return dev_data
