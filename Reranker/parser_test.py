import os
import dev_reader
import data_util
from eval import eval as eval_tool
import numpy as np
import random
import dependency_model

DIR = 'd:\\MacShare\\data2\\'
TRAIN = 'train'
DEV = 'dev'
TEST = 'test'
OUTPUT_MODEL = 'model_best.pkl'
OUTPUT_DICT = 'dict.pkl'
PAIR_WISE = False

def test_model():
    max_degree,vocab = data_util.load_dict(os.path.join(DIR, OUTPUT_DICT))
    dev_data = dev_reader.read_dev(os.path.join(DIR, DEV + '.kbest'),
                                            os.path.join(DIR, DEV + '.gold'), vocab)
    # test_data = dev_reader.read_dev(os.path.join(DIR, TEST + '.kbest'),
    #                                     os.path.join(DIR, TEST + '.gold'), vocab)
    #evaluate_oracle_worst(test_data)
    evaluate_baseline_random(dev_data)
    evaluate_oracle_worst(dev_data)
    print 'build model'
    model = dependency_model.get_model(vocab.size(),vocab.tagsize(), max_degree,PAIR_WISE)
    print 'load params'
    model.set_parmas(os.path.join(DIR,OUTPUT_MODEL))
    if model.Pairwise:
        evaluate_dataset_pair(model,dev_data)
    else:
        evaluate_dataset_point(model, dev_data,False)
    #evaluate_dataset(model, test_data, True)


def evaluate_baseline(data):
    pred_trees = []
    gold_trees = []
    for i, inst in enumerate(data):
        for line in inst.lines[len(inst.kbest)-1]:
            pred_trees.append(line)
        pred_trees.append('\n')
        for line in inst.gold_lines:
            gold_trees.append(line)
        gold_trees.append('\n')
    print 'baseline: %.4f' % (eval_tool.evaluate(pred_trees,gold_trees)[0])

def evaluate_baseline_random(data):
    random.seed(189)
    pred_trees = []
    gold_trees = []
    for i, inst in enumerate(data):
        rand = int(random.random()*len(inst.kbest))
        for line in inst.lines[rand]:
            pred_trees.append(line)
        pred_trees.append('\n')
        for line in inst.gold_lines:
            gold_trees.append(line)
        gold_trees.append('\n')
    print 'baseline: %.4f' % (eval_tool.evaluate(pred_trees,gold_trees)[0])


def evaluate_oracle_worst(data):
    oracle_trees = []
    worst_trees = []
    gold_trees = []
    pred_trees = []
    for i, inst in enumerate(data):
        max = 0
        maxid = 0
        min = 1
        minid = 0
        for line in inst.gold_lines:
            gold_trees.append(line)
        gold_trees.append('\n')

        for line in inst.lines[len(inst.kbest)-1]:
            pred_trees.append(line)
        pred_trees.append('\n')

        i = 0
        for list in inst.lines:
            temp = []
            for line in list:
                temp.append(line)
            temp.append('\n')
            res = eval_tool.evaluate(temp, inst.gold_lines)[0]
            if res > max :
                max = res
                maxid = i
            if res < min :
                min = res
                minid = i
            i += 1
        for line in inst.lines[maxid]:
            oracle_trees.append(line)
        oracle_trees.append('\n')
        for line in inst.lines[minid]:
            worst_trees.append(line)
        worst_trees.append('\n')
    print 'f1score: %.4f'  % (eval_tool.evaluate(pred_trees, gold_trees)[0])
    print 'oracle: %.4f'  % (eval_tool.evaluate(oracle_trees, gold_trees)[0])
    print 'worst: %.4f'  % (eval_tool.evaluate(worst_trees, gold_trees)[0])

def evaluate_dataset_point(model, data , addbase):
    pred_trees = []
    gold_trees = []
    for i, inst in enumerate(data):
        pred_scores = []
        for tree in inst.kbest:
            if tree.size == inst.gold.size:
                pred_scores.append(model.predict(tree))
            else:
                print 'error'
                pred_scores.append(-1000)
        #data_util.normalize(pred_scores)
        if addbase:
            data_util.normalize(inst.scores)
            scores = [p_s + b_s for p_s, b_s in zip(pred_scores, inst.scores)]
        else:
            scores = pred_scores
        max_id = scores.index(min(scores))
        #print max_id,scores[max_id]
        for line in inst.lines[max_id]:
            pred_trees.append(line)
        pred_trees.append('\n')
        for line in inst.gold_lines:
            gold_trees.append(line)
        gold_trees.append('\n')
    res = eval_tool.evaluate(pred_trees,gold_trees)
    print 'f1score: %.4f' % (res[0])
    return res

def evaluate_dataset_pair(model, data):
    pred_trees = []
    gold_trees = []
    for i, inst in enumerate(data):
        lens = len(inst.kbest)
        max = 0
        for j in range(1, lens):
            loss = 0
            if inst.kbest[j].size == inst.gold.size:
                loss = np.mean(model.train_pairwise(inst.kbest[max],inst.kbest[j],True))
            if loss > 0:
                max = j
        for line in inst.lines[max]:
            pred_trees.append(line)
        pred_trees.append('\n')
        for line in inst.gold_lines:
            gold_trees.append(line)
        gold_trees.append('\n')
    res = eval_tool.evaluate(pred_trees,gold_trees)
    print 'f1score: %.4f' % (res[0])
    return res
if __name__ == '__main__':
    test_model()
