import torch
from torch import nn
import re

# 数据预处理#
START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {"B": 0, "I": 1, "E": 2, "S": 3, START_TAG: 4, STOP_TAG: 5}


def prepare_sequence(seq, to_ix):  # seq是字序列，to_ix是字和序号的字典
    idxs = [to_ix[w] for w in seq]  # idxs是字序列对应的向量
    return torch.tensor(idxs, dtype=torch.long)


def read_file(filename):
    content, label = [], []
    text = open(filename, 'r', encoding='utf-8')
    for eachline in text:
        if (eachline in ['\n','\r\n']):
            continue
        line = eachline.split(' ')
        content.append(line[0])
        label.append(line[1])
    return content, label


# 实现model#

def argmax(vec):  # 返回每一行最大值的索引
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):  # seq是字序列，to_ix是字和序号的字典
    idxs = [to_ix[w] for w in seq]  # idxs是字序列对应的向量
    return torch.tensor(idxs, dtype=torch.long)


# LSE函数，模型中经常用到的一种路径运算的实现
def log_sum_exp(vec):  # vec.shape=[1, target_size]
    max_score = vec[0, argmax(vec)]  # 每一行的最大值
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)  # Maps the output of the LSTM into tag space

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))  # 随机初始化转移矩阵

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000  # tag_to_ix[START_TAG]: 3（第三行，即其他状态到START_TAG的概率）
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000  # tag_to_ix[STOP_TAG]: 4（第四列，即STOP_TAG到其他状态的概率）
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

    # 所有路径的得分，CRF的分母
    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)  # 初始隐状态概率，第1个字是O1的实体标记是qi的概率
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas  # 初始状态的forward_var，随着step t变化

        for feat in feats:  # feat的维度是[1, target_size]
            alphas_t = []
            for next_tag in range(self.tagset_size):  # 给定每一帧的发射分值，按照当前的CRF层参数算出所有可能序列的分值和

                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)  # 发射概率[1, target_size] 隐状态到观测状态的概率
                trans_score = self.transitions[next_tag].view(1, -1)  # 转移概率[1, target_size] 隐状态到下一个隐状态的概率
                next_tag_var = forward_var + trans_score + emit_score  # 本身应该相乘求解的，因为用log计算，所以改为相加

                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[
            self.tag_to_ix[STOP_TAG]]  # 最后转到[STOP_TAG]，发射分值为0，转移分值为列向量 self.transitions[:, [self.tag2ix[END_TAG]]]
        return log_sum_exp(terminal_var)

    # 得到feats，维度=len(sentence)*tagset_size，表示句子中每个词是分别为target_size个tag的概率
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # 正确路径的分数，CRF的分子
    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]] 是从标签i到标签i+1的转移概率
            # feat[tags[i+1]], feat是step i的输出结果，有５个值，对应B, I, E, START_TAG, END_TAG, 取对应标签的值
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]  # 沿途累加每一帧的转移和发射
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]  # 加上到END_TAG的转移
        return score

    # 解码，得到预测序列的得分，以及预测的序列
    def _viterbi_decode(self, feats):
        backpointers = []  # 回溯路径；backpointers[i][j]=第i帧到达j状态的所有路径中, 得分最高的那条在i-1帧是什么状态

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]  # 其他标签（B,I,E,Start,End）到标签next_tag的概率
                best_tag_id = argmax(next_tag_var)  # 选择概率最大的一条的序号
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)  # 从step0到step(i-1)时5个序列中每个序列的最大score
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]  # 其他标签到STOP_TAG的转移概率
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):  # 从后向前走，找到一个best路径
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # 安全性检查
        best_path.reverse()  # 把从后向前的路径倒置
        return path_score, best_path

    # 求负对数似然，作为loss
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)  # emission score
        forward_score = self._forward_alg(feats)  # 所有路径的分数和，即b
        gold_score = self._score_sentence(feats, tags)  # 正确路径的分数，即a
        return forward_score - gold_score  # 注意取负号 -log(a/b) = -[log(a) - log(b)] = log(b) - log(a)

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
