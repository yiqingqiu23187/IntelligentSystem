from wordseg.BI_LSTM_CRF import *
from wordseg.data_process import *
import torch
from torch import optim

# 超参数#
filename = 'D:/PJ/IntelliLabs/Lab2/dataset2/train.utf8'
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
epochs = 10
##


content, label = read_file(filename)


def train_data(content, label):
    train_data = []
    for i in range(len(label)):
        train_data.append((content[i], label[i]))
    return train_data


data = train_data(content, label)

word_to_ix = {}
for sentence, tags in data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)  # 单词映射，字到序号

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练
print("training...")
for epoch in range(epochs):
    print("training in epoch"+str(epoch))
    loss = 0
    for sentence, tags in data:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        loss = model.neg_log_likelihood(sentence_in, targets)

        loss.backward()
        optimizer.step()

    # 保存模型
    torch.save(model, '../LSTMmodel/version2_model' + str(epoch))
    torch.save(model.state_dict(), '../LSTMmodel/version2_cws_all.model' + str(epoch))
    print('epoch/epochs: {}/{}, loss:{:.6f}'.format(epoch + 1, epochs, loss.data[0]))

# 保存模型
# torch.save(model,'../LSTMmodel/version2_model0')
# torch.save(model.state_dict(),'../LSTMmodel/version2_cws_all.model0')
