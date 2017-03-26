import pickle
import math
from eval import eval as eval_tool
import Vocab

class instance(object):
    def __init__(self,kbest,scores,gold,lines,gold_lines):
        self.kbest = kbest
        self.scores = scores
        self.gold = gold
        self.gold_lines = gold_lines
        self.lines = lines
        self.f1score = []
        #self.maxid = self.get_oracle_index()

    def set_f1(self):
        for l in self.lines:
            f1 = eval_tool.evaluate(l, self.gold_lines)[0]
            self.f1score.append(f1)

    def get_oracle_index(self):
        max = 0
        maxid = 0
        i = 0
        for list in self.lines:
            temp = []
            for line in list:
                temp.append(line)
            temp.append('\n')
            res = eval_tool.evaluate(temp, self.gold_lines)[0]
            self.f1score.append(res)
            if res > max:
                max = res
                maxid = i
            i += 1
        return maxid

def normalize(list):
    sum = 0
    max_score = max(list)
    for i in range(len(list)):
        list[i] = math.pow(1.1, list[i]-max_score)
        sum += list[i]

    for i in range(len(list)):
        list[i] = math.log(list[i]/sum, 1.1)



def save_model(model,output_file):
    output = open(output_file, 'wb')
    for shared_value in model.params:
        pickle.dump(shared_value.get_value(),output, protocol=2)
    output.close()


def save_dict(vocab,degree,output_file):
    output = open(output_file, 'wb')
    pickle.dump(degree,output, protocol=2)
    pickle.dump(vocab,output, protocol=2)
    output.close()

def load_dict(input_file):
    pkl_file = open(input_file, 'rb')
    degree = pickle.load(pkl_file)
    dict = pickle.load(pkl_file)
    pkl_file.close()
    return degree,dict

def load_model(input_file):
    pkl_file = open(input_file, 'rb')
    params = pickle.load(pkl_file)
    pkl_file.close()
    return params