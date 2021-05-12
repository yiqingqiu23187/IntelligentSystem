class HmmModel:
    state_list = ['B', 'I', 'E', 'S']
    line_num = -1
    INPUT_DATA = "D:/PJ/IntelliLabs/Lab2/dataset1/train.utf8"

    def __init__(self, input_data):
        self.A_dic = {}  # 转移概率矩阵
        self.B_dic = {}  # 发射概率矩阵
        self.Count_dic = {}
        self.Pi_dic = {}  # 初始状态分布
        self.INPUT_DATA = input_data
        self.train()

    def init(self):  # 初始化字典
        for state in self.state_list:
            self.A_dic[state] = {}
            for state1 in self.state_list:
                self.A_dic[state][state1] = 0.0
        for state in self.state_list:
            self.Pi_dic[state] = 0.0
            self.B_dic[state] = {}
            self.Count_dic[state] = 0

    # 输出模型的三个参数：初始概率+转移概率+发射概率
    def output(self):
        for key in self.Pi_dic:  # 状态的初始概率
            self.Pi_dic[key] = self.Pi_dic[key] * 1.0 / self.line_num

        for key in self.A_dic:  # 状态转移概率
            for key1 in self.A_dic[key]:
                self.A_dic[key][key1] = self.A_dic[key][key1] / self.Count_dic[key]

        for key in self.B_dic:  # 发射概率(状态->词语的条件概率)
            for word in self.B_dic[key]:
                self.B_dic[key][word] = self.B_dic[key][word] / self.Count_dic[key]

    def train(self):
        self.init()
        ifp = open(self.INPUT_DATA, encoding="utf8")
        word_list = []
        line_state = []
        for line in ifp:
            line = line.strip()
            if not line:
                self.line_num += 1
                for i in range(len(line_state)):
                    if i == 0:
                        self.Pi_dic[line_state[0]] += 1  # Pi_dic记录句子第一个字的状态，用于计算初始状态概率
                        self.Count_dic[line_state[0]] += 1  # 记录每一个状态的出现次数
                    else:
                        self.A_dic[line_state[i - 1]][line_state[i]] += 1  # 用于计算转移概率
                        self.Count_dic[line_state[i]] += 1
                        if not word_list[i] in self.B_dic[line_state[i]]:
                            self.B_dic[line_state[i]][word_list[i]] = 0.0

    def decode(self, sequence):
        """
        Decode the given sequence.
        """
        sequence_length = len(sequence)

        delta = {}
        for state in self.state_list:
            delta[state] = self.Pi_dic[state] * self.B_dic[state][sequence[0]]

        pre = []
        for index in range(1, sequence_length):
            delta_bar = {}
            pre_state = {}
            for state_to in self.state_list:
                max_prob = 0
                max_state = None
                for state_from in self.state_list:
                    prob = delta[state_from] * self.A_dic[state_from][state_to]
                    if prob > max_prob:
                        max_prob = prob
                        max_state = state_from
                delta_bar[state_to] = max_prob * self.B_dic[state_to][sequence[index]]
                pre_state[state_to] = max_state
            delta = delta_bar
            pre.append(pre_state)

        max_state = None
        max_prob = 0
        for state in self.state_list:
            if delta[state] > max_prob:
                max_prob = delta[state]
                max_state = state

        if max_state is None:
            return []

        result = [max_state]
        for index in range(sequence_length - 1, 0, -1):
            max_state = pre[index - 1][max_state]
            result.insert(0, max_state)

        return result


if __name__ == "__main__":
    hmm = HmmModel("D:/PJ/IntelliLabs/Lab2/dataset1/train.utf8")
    hmm.train()


def viterbi(self, obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    for y in states:  # 初始值
        V[0][y] = start_p[y] * emit_p[y].get(obs[0], 0)  # 在位置0，以y状态为末尾的状态序列的最大概率
        path[y] = [y]
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:  # 从y0 -> y状态的递归
            (prob, state) = max(
                [(V[t - 1][y0] * trans_p[y0].get(y, 0) * emit_p[y].get(obs[t], 0), y0) for y0 in states])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath  # 记录状态序列
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])  # 在最后一个位置，以y状态为末尾的状态序列的最大概率
    res_str = ''.join(path[state])
    return res_str