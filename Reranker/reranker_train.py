import os

import numpy as np
import parser_test
import data_reader
import data_util
import dependency_model
import random
USE_BATCH = True
DIR = 'd:\\MacShare\\data2\\'
TRAIN = 'train'
DEV = 'dev'
TEST = 'test'
OUTPUT_MODEL = 'model.pkl'
OUTPUT_BEST = 'model_best_pairwise.pkl'
OUTPUT_DICT = 'dict.pkl'
TRAIN_BATCH_SIZE = 100
PAIR_WISE = False
NUM_EPOCHS = 20
SEED = 88

def train_dataset(model, data, echo,batch,pairwise):
    losses = []
    avg_loss = 0.0
    total_data = len(data)
    loss = 0.0
    print 'echo %s batch %d' % (echo, batch)
    for i, inst in enumerate(data):
        if pairwise:
            loss = model.train_step_pairwise(inst)
        else:
            loss = model.train_step_pointwise(inst.kbest, inst.gold)
        losses.append(loss)
        #print 'echo %s batch %d size %d instance: %d  loss: %.10f' %(echo, batch ,len(data),i,loss)
        #avg_loss = avg_loss * (len(losses) - 1) / len(losses) + loss / len(losses)
        #print 'echo %d batch %d avg loss %.4f example id %d batch size %d\r' % (echo ,batch,avg_loss, inst.id, total_data)
    loss = np.mean(losses)
    print 'loss %.10f' % loss
    return loss

def train_model():
    data_tool = data_reader.data_manager(TRAIN_BATCH_SIZE,os.path.join(DIR,TRAIN+'.kbest'),
                         os.path.join(DIR,TRAIN+'.gold'),
                         os.path.join(DIR, DEV + '.kbest'),
                         os.path.join(DIR, DEV + '.gold'),vocab_path= os.path.join(DIR, OUTPUT_DICT))


    dev_data = data_tool.dev_data
    np.random.seed(SEED)
    print 'build model'
    model = dependency_model.get_model(data_tool.vocab.size(),data_tool.vocab.tagsize(), data_tool.max_degree,PAIR_WISE)
    print 'model established'
    max_uas = 0
    if PAIR_WISE:
        parser_test.evaluate_dataset_pair(model, dev_data)
    else:
        parser_test.evaluate_dataset_point(model, dev_data,False)
    for i in range(NUM_EPOCHS):
        print 'Echo %d train' % (i)
        data = data_tool.train_iter.read_batch()
        batchid = 0
        while not data is None:
            if PAIR_WISE:
                for inst in data:
                    inst.set_f1()
            train_dataset(model, data ,i,batchid,PAIR_WISE)
            data = data_tool.train_iter.read_batch()
            batchid += 1
        data_tool.train_iter.reset()
        if PAIR_WISE:
            uas = parser_test.evaluate_dataset_pair(model, dev_data)[0]
        else:
            uas = parser_test.evaluate_dataset_point(model, dev_data, False)[0]
        if uas > max_uas:
            max_uas = uas
            data_util.save_model(model, os.path.join(DIR,OUTPUT_BEST))
        data_util.save_model(model, os.path.join(DIR, OUTPUT_MODEL))
    print 'best score %.4f' % max_uas

if __name__ == '__main__':
    train_model()
