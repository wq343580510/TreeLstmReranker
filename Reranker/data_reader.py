import os
import Vocab
import data_util
import dev_reader
import train_iterator



class data_manager(object):
    max_degree = 0
    def __init__(self,batch,train_kbest = None,train_gold = None,dev_kbest = None,dev_gold = None,
                 test_kbest = None,test_gold = None,vocab_path = None):
        self.vocab = None
        self.train_kbest = train_kbest
        self.train_gold = train_gold
        self.dev_kbest = dev_kbest
        self.dev_gold = dev_gold
        self.batch = batch
        self.test_kbest = test_kbest
        self.test_gold = test_gold
        if os.path.exists(vocab_path):
            print 'load vocab'
            self.max_degree,self.vocab = data_util.load_dict(vocab_path)
        else:
            print 'creat vocab'
            self.vocab = Vocab.Vocab(self.train_gold)
            print 'get max_degree'
            self.max_degree = self.get_max_degree()
            print 'save dictionary'
            data_util.save_dict(self.vocab,self.max_degree, vocab_path)
        print 'vocab size:' + str(self.vocab.size())
        print 'max_degree' + str(self.max_degree)
        print 'get dev data'
        self.dev_data = dev_reader.read_dev(dev_kbest,dev_gold,self.vocab)
        print 'number of dev:'+str(len(self.dev_data))
        #self.test_data = dev_reader.read_dev(test_kbest,test_gold,self.vocab)
        print 'create train batch'
        self.train_iter = train_iterator.train_iterator(train_kbest,train_gold,self.vocab,self.batch)

        # print 'get train data'
        # self.train_data = dev_reader.read_dev(train_kbest,train_gold,self.vocab)
        # print 'number of train:'+str(len(self.train_data))

    def get_max_degree(self):
        retval = 0
        for file in [self.train_kbest, self.train_gold, self.dev_gold, self.dev_kbest]:
            f = open(file)
            line = f.readline()
            list = []
            while line:
                if line.strip() != 'PTB_KBEST' and '_' in line:
                    list.append(line)
                elif line.strip() == '':
                    parents = {}
                    for s in list:
                        p = int(s.split()[6])
                        if parents.has_key(p):
                            parents[p] += 1
                        else:
                            parents[p] = 1
                    max_degree = max(parents.values())
                    if max_degree > retval:
                        retval = max_degree
                    list = []

                line = f.readline()
        return retval






