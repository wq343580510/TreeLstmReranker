import data_reader
import data_util
import tree_rnn
DIR = 'd:\\MacShare\\data\\'
TRAIN = 'train'
DEV = 'dev'
TEST = 'test'

class train_iterator(object):
    def __init__(self, kbest_filename , gold_filename, vocab,batch):
        with open(kbest_filename, 'r') as reader:
            self.data = reader.readlines()
            self.data.append('PTB_KBEST')
        with open(gold_filename, 'r') as reader:
            self.gdata = reader.readlines()
        self.kbest_id = 0
        self.vocab = vocab
        self.index = 0
        self.gindex = 0
        self.batch = batch
        self.length = len(self.data)
        self.glength = len(self.gdata)

    def read_give_tree(self,tree_index):
        scores = []
        tree = []
        ktrees = []
        kbestlines = []
        # read train
        index = 0
        best_num = 0
        while index < self.length:
            line = self.data[index]
            if best_num == tree_index:
                if line.strip() != 'PTB_KBEST':
                    if line.strip() == '':
                        ktrees.append(read_tree(tree, self.vocab))
                        kbestlines.append(tree[:])
                        tree = []
                    elif not '_' in line:
                        scores.append(float(line))
                    else:
                        tree.append(line)
                else:
                    break
            if line.strip() == 'PTB_KBEST':
                best_num += 1
            index += 1
        # read gold
        list = []
        gold = []
        index = 0
        num = 1
        while index < self.glength:
            line = self.gdata[index]
            if num == tree_index:
                if line.strip() == '':
                    root = read_tree(list, self.vocab)
                    gold.append(root)
                    break
                else:
                    list.append(line)
            if line.strip() == '':
                num += 1
            index += 1

        retval = data_util.instance(ktrees, scores, gold,gold_lines=list,lines=kbestlines)
        return retval

    def read_all(self):
        scores = []
        kscores = []
        tree = []
        ktrees = []
        kbest = []
        lines = []
        klines = []
        index = 0
        while index < self.length:
            line = self.data[index]
            if line.strip() != 'PTB_KBEST':
                if line.strip() == '':
                    ktrees.append(read_tree(tree, self.vocab))
                    lines.append(tree[:])
                    tree = []
                elif not '_' in line:
                    scores.append(float(line))
                else:
                    tree.append(line)
            else:
                if len(ktrees) > 2:
                    kbest.append(ktrees[:])
                    kscores.append(scores[:])
                    klines.append(lines[:])
                    lines = []
                    ktrees = []
                    scores = []
            index += 1
        # read gold
        list = []
        gold = []
        goldlines = []
        gindex =  0
        while gindex < self.glength:
            line = self.gdata[gindex]
            if line.strip() == '':
                root = read_tree(list, self.vocab)
                goldlines.append(list[:])
                gold.append(root)
                list = []
            else:
                list.append(line)
            gindex += 1
        train_batch = []
        for a, b, c , d, e in zip(kbest, kscores, gold,klines,goldlines):
            self.kbest_id += 1
            if len(c.children) == 0:
                continue
            train_batch.append(data_util.instance(a, b, c, d,e))
        return train_batch

    def read_random_batch(self, batch_size = 400):
        total_list = []
        for i in range(39830):
            total_list.append(i)
        import random
        random.shuffle(total_list)
        total_list = total_list[0:batch_size]
        sorted(total_list)
        scores = []
        kscores = []
        tree = []
        ktrees = []
        kbest = []
        # read train
        lines = []
        klines = []
        i = 0
        while i < self.length:
            line = self.data[i]
            if line.strip() != 'PTB_KBEST':
                if line.strip() == '':
                    ktrees.append(read_tree(tree, self.vocab))
                    lines.append(tree[:])
                    tree = []
                elif not '_' in line:
                    scores.append(float(line))
                else:
                    tree.append(line)
            else:
                if len(ktrees) > 2:
                    kbest.append(ktrees[:])
                    kscores.append(scores[:])
                    klines.append(lines[:])
                    if read_batch and len(kbest) == self.batch:
                        self.index += 1
                        break
                    ktrees = []
                    scores = []
                    lines = []
            self.index += 1
        # read gold
        list = []
        gold = []
        goldlines = []
        while self.gindex < self.glength:
            line = self.gdata[self.gindex]
            if line.strip() == '':
                root = read_tree(list, self.vocab)
                gold.append(root)
                goldlines.append(list[:])
                if read_batch and len(gold) == self.batch:
                    self.gindex += 1
                    break
                list = []
            else:
                list.append(line)
            self.gindex += 1
        train_batch = []
        for a, b, c, d, e in zip(kbest, kscores, gold, klines, goldlines):
            self.kbest_id += 1
            if len(c.children) == 0:
                continue
            train_batch.append(data_util.instance(a, b, c, d, e))
        return train_batch

    def read_batch(self,read_batch = True):
        if self.index == self.length:
            return None
        scores = []
        kscores = []
        tree = []
        ktrees = []
        kbest = []
        #read train
        lines = []
        klines = []
        while self.index < self.length:
            line = self.data[self.index]
            if line.strip() != 'PTB_KBEST':
                if line.strip() == '':
                    ktrees.append(read_tree(tree,self.vocab))
                    lines.append(tree[:])
                    tree = []
                elif not '_' in line:
                    scores.append(float(line))
                else:
                    tree.append(line)
            else:
                if len(ktrees) > 2:
                    kbest.append(ktrees[:])
                    kscores.append(scores[:])
                    klines.append(lines[:])
                    if read_batch and len(kbest) == self.batch:
                        self.index += 1
                        break
                    ktrees = []
                    scores = []
                    lines = []
            self.index += 1
        #read gold
        list = []
        gold = []
        goldlines = []
        while self.gindex < self.glength:
            line = self.gdata[self.gindex]
            if line.strip() == '':
                root = read_tree(list,self.vocab)
                gold.append(root)
                goldlines.append(list[:])
                if read_batch and len(gold) == self.batch:
                    self.gindex += 1
                    break
                list = []
            else:
                list.append(line)
            self.gindex += 1
        train_batch = []
        for a,b,c,d,e in zip(kbest,kscores,gold,klines,goldlines):
            self.kbest_id += 1
            if len(c.children) == 0:
                continue
            train_batch.append(data_util.instance(a,b,c,d,e))
        return train_batch
    def reset(self):
        self.index = 0
        self.gindex = 0
        self.kbest_id = 0

def read_tree(list,vocab):
    att_list = []
    nodes = []
    root = None
    for i in range(len(list)):
        att_list.append(list[i].split())
        word = att_list[i][1]
        tag = att_list[i][3]
        if vocab is None:
            val = word
            tag_idx = 0
        else:
            val = vocab.index(word)
            tag_idx = vocab.indexoftag(tag)
        nodes.append(tree_rnn.Node(val,i,tag_idx))
    for i in range(len(list)):
        parent = int(att_list[i][6]) - 1
        if parent >= 0:
            nodes[parent].add_child(nodes[i])
        elif parent == -1:
            root = nodes[i]
    return root
