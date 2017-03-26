import pickle
import theano
from theano import tensor as T
import numpy as np
import data_util
import tree_rnn
import data_reader

LEARNING_RATE = 0.1
EMB_DIM = 50
HIDDEN_DIM = 200
OUTPUT_DIM = 1


class DependencyModel(tree_rnn.TreeRNN):
    # def set_parmas(self,w,b):
    #     self.W_u.set_value(w)
    #     self.W_i.set_value(w)
    #     self.W_o.set_value(w)
    #     self.W_f.set_value(w)
    #     self.W_gate.set_value(w)
    #     self.U_u.set_value(w)
    #     self.U_i.set_value(w)
    #     self.U_o.set_value(w)
    #     self.U_f.set_value(w)
    #     self.W_out.set_value(w)
    #     self.b_u.set_value(b)
    #     self.b_i.set_value(b)
    #     self.b_o.set_value(b)
    #     self.b_f.set_value(b)
    #     self.W_out.set_value(w)
    #     self.b_out.set_value(b)
    def set_parmas(self, input_file):
        pkl_file = open(input_file, 'rb')
        self.embeddings.set_value(pickle.load(pkl_file))
        self.W_i.set_value(pickle.load(pkl_file))
        self.U_i.set_value(pickle.load(pkl_file))
        self.b_i.set_value(pickle.load(pkl_file))
        self.W_f.set_value(pickle.load(pkl_file))
        self.U_f.set_value(pickle.load(pkl_file))
        self.b_f.set_value(pickle.load(pkl_file))
        self.W_o.set_value(pickle.load(pkl_file))
        self.U_o.set_value(pickle.load(pkl_file))
        self.b_o.set_value(pickle.load(pkl_file))
        self.W_u.set_value(pickle.load(pkl_file))
        self.U_u.set_value(pickle.load(pkl_file))
        self.b_u.set_value(pickle.load(pkl_file))
        self.scoreVector.set_value(pickle.load(pkl_file))
        if self.Pairwise:
            self.W_gate.set_value(pickle.load(pkl_file))
            self.U_gate.set_value(pickle.load(pkl_file))
            self.b_gate.set_value(pickle.load(pkl_file))
        pkl_file.close()

    def set_emb(self,emb):
        self.embeddings.set_value(emb)

    def train_step_pointwise(self, kbest_tree, gold_root):
        scores = []
        for tree in kbest_tree:
            if tree.size == gold_root.size:
                scores.append(self.predict(tree))
            else:
                scores.append(min(scores))
        max_id = scores.index(max(scores))
        pred_root = kbest_tree[max_id]
        if pred_root.size != gold_root.size:
            return 0
        gold_score = self.predict(gold_root)
        pred_score = scores[max_id]
        loss = pred_score - gold_score
        if loss > 0:
            self.train_pointwise(pred_root,gold_root)
        return loss

    def train_step_pairwise(self,inst):
        kbest_tree = inst.kbest
        local_best = kbest_tree[0]
        best_f1 = 0
        loss = 0
        for i in range(1,len(kbest_tree)):
            if kbest_tree[i].size != inst.gold.size:
                continue
            if inst.f1score[i] > best_f1:
                subloss = np.mean(self.train_pairwise(kbest_tree[i],local_best,False))
                best_f1 = inst.f1score[i]
                local_best = kbest_tree[i]
            else:
                subloss = np.mean(self.train_pairwise(local_best,kbest_tree[i],False))
            if subloss > 0:
                loss += subloss
        return loss

def get_model(num_emb, tag_size, max_degree,p):
    return DependencyModel(
        num_emb,tag_size, EMB_DIM, HIDDEN_DIM, OUTPUT_DIM,
        degree=max_degree, learning_rate=LEARNING_RATE,
        trainable_embeddings=True,pairwise = p
        )

