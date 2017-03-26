import os
import dev_reader
import data_util
import dependency_model
import  numpy as np
from eval import eval as eval_tool

DIR = 'd:\\MacShare\\data2\\'
TRAIN = 'train'
DEV = 'dev'
TEST = 'test'
OUTPUT_MODEL = 'model_best_pairwise.pkl'
OUTPUT_DICT = 'dict.pkl'
PAIR_WISE = True

def test_model():
    max_degree,vocab = data_util.load_dict(os.path.join(DIR, OUTPUT_DICT))
    dev_data = dev_reader.read_dev(os.path.join(DIR, DEV + '.kbest'),
                                            os.path.join(DIR, DEV + '.gold'), vocab)
    test_data = dev_reader.read_dev(os.path.join(DIR, TEST + '.kbest'),
                                             os.path.join(DIR, TEST + '.gold'), vocab)
    print 'model file name %s' % OUTPUT_MODEL
    print 'build model'
    #model = dependency_model.get_model(vocab.size(),vocab.tagsize(),max_degree,PAIR_WISE)
    print 'load params'
    #model.set_parmas(os.path.join(DIR,OUTPUT_MODEL))
    model = 0
    max = 0
    max_r = 0
    for i in range(200):
        if PAIR_WISE:
            res = evaluate_dataset_pair(model,dev_data,True,ratio=0.005*i)
        else:
            res = evaluate_dataset_point(model,dev_data,True,ratio=0.005*i)
        if res[0]>max:
            max = res[0]
            max_r = 0.005*i
    print max_r,max

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


def evaluate_dataset_pair(model, data,addbase ,ratio = 1):
    pred_trees = []
    gold_trees = []
    for i, inst in enumerate(data):
        lens = len(inst.kbest)
        max = 0
        for j in range(1, lens):
            loss = 0
            if inst.kbest[j].size == inst.gold.size:
                #loss = np.mean(model.train_pairwise(inst.kbest[max],inst.kbest[j],True))
                # score_j - score_best
                if addbase:
                    baseloss = inst.scores[j]-inst.scores[max]
                    #print "loss: %.4f    baseloss: %.4f" % (loss,baseloss)
                    loss += baseloss
            if loss > 0:
                max = j
        for line in inst.lines[max]:
            pred_trees.append(line)
        pred_trees.append('\n')
        for line in inst.gold_lines:
            gold_trees.append(line)
        gold_trees.append('\n')
    res = eval_tool.evaluate(pred_trees,gold_trees)
    print 'ratio: %f f1score: %.4f' % (ratio,res[0])
    return res

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


def evaluate_dataset_point(model, data , addbase ,ratio = 1):
    pred_trees = []
    gold_trees = []
    for i, inst in enumerate(data):
        pred_scores = [model.predict(tree) for tree in inst.kbest if tree.size == inst.gold.size]
        data_util.normalize(pred_scores)
        if addbase:
            data_util.normalize(inst.scores)
            scores = [ratio*p_s + (1-ratio)*b_s for p_s, b_s in zip(pred_scores, inst.scores)]
        else:
            scores = pred_scores
        max_id = scores.index(max(scores))
        #print "pred: %.4f    base: %.4f" % (pred_scores[max_id],inst.scores[max_id])
        for line in inst.lines[max_id]:
            pred_trees.append(line)
        pred_trees.append('\n')
        for line in inst.gold_lines:
            gold_trees.append(line)
        gold_trees.append('\n')
    res = eval_tool.evaluate(pred_trees,gold_trees)
    print 'ratio: %f f1score: %.4f' % (ratio,res[0])
    return res
if __name__ == '__main__':
    test_model()
