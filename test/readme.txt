这几个文件不需要看
eval_ratio oracle 
test 
test_error_tree
train_data_iter
train_iterator


data_reader.py:
	data_manager主要是建立词典，获得最大的度（tree_lstm建立model时会用到），
				读取train和dev
data_util.py
	instance 表示一个训练样本
	还有一些功能型的函数

dev_reader.py 读取dev集合的类，由于dev data需要算f1值而train data不需要
所以有点不一样。

主要稍微复杂的代码在
tree_rnn.py里面
这个是实现了tree_rnn model，然后tree_lstm是在它上面的扩展，
我们用的dependency_model则是在tree_lstm的子类。
tree_lstm的NaryTreeLSTM不用管。

程序的入口是在reranker_train.py
唯一有点复杂的地方在tree_rnn.py里面
tree_lstm是tree_rnn.py的子类。

下面是tree_rnn的源代码，我是在他的基础上改了一些东西
https://github.com/ofirnachum/tree_rnn