import pickle
import torch


class LSTMModel:
    def predict(self, sentence):
        model = torch.load('wordseg/model/model017.pkl')
        # model = torch.load('model10.pkl')
        with open('wordseg/datasave1.pkl', 'rb') as inp:
            word2id = pickle.load(inp)
            id2word = pickle.load(inp)
            tag2id = pickle.load(inp)
            id2tag = pickle.load(inp)
        inp.close()
        length = len(id2word)
        x = []
        for word in sentence:
            if (word in id2word):
                x.append(word2id[word])
            else:
                x.append(length - 1)
        sentence = torch.tensor(x, dtype=torch.long)
        score, predict = model.test(sentence)
        y = []
        for p in predict:
            y.append(id2tag[p])
        y = ''.join(y)
        return y
