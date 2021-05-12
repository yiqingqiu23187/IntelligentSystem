import  pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from wordseg.LSTM import Model

def calculate(x,y,id2word,id2tag,res=[]):
    entity=[]
    for j in range(len(x)):
        if id2tag[y[j]]=='B':
            entity=[id2word[x[j]]]
        elif id2tag[y[j]]=='I' and len(entity)!=0:
            entity.append(id2word[x[j]])
        elif id2tag[y[j]]=='E' and len(entity)!=0:
            entity.append(id2word[x[j]])
            res.append(entity)
            entity=[]
        elif id2tag[y[j]]=='S':
            entity=[id2word[x[j]]]
            res.append(entity)
            entity=[]
        else:
            entity=[]
    return res


with open('../LSTMmodel/datasave.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
    x_train = pickle.load(inp)
    y_train = pickle.load(inp)
    x_test = pickle.load(inp)
    y_test = pickle.load(inp)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
EPOCHS = 10
LR=0.005
tag2id[START_TAG]=len(tag2id)
tag2id[STOP_TAG]=len(tag2id)

model = Model(len(word2id) + 1, tag2id, EMBEDDING_DIM, HIDDEN_DIM)

optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-4)



for epoch in range(EPOCHS):
    index = 0
    for sentence, tags in zip(x_train, y_train):
        index += 1
        model.zero_grad()

        sentence = torch.tensor(sentence, dtype=torch.long)
        tags = torch.tensor(tags, dtype=torch.long)

        loss = model(sentence, tags)

        loss.backward()
        optimizer.step()
        if index % 10000 == 0:
            print("epoch", epoch, "index", index)
    entityres = []
    entityall = []
    path_name = "../LSTMmodel/lstmmodel" + str(epoch) + ".pkl"
    torch.save(model, path_name)
    print("model has been saved in  ", path_name)
# path_name = "../LSTMmodel/lstmmodel" + str(0) + ".pkl"
# torch.save(model, path_name)
# print("model has been saved in  ", path_name)