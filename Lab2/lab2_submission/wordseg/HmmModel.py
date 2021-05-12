class HmmModel:
    state_list = ['B', 'I', 'E', 'S']
    line_num = -1
    #INPUT_DATA = "../../dataset1/train.utf8"

    def __init__(self, input_data):
        self.trans_p = {}  # 转移概率矩阵
        self.emit_p = {}  # 发射概率矩阵
        self.Count_dic = {}
        self.initial_p = {}  # 初始状态分布
        self.INPUT_DATA = input_data

        self.train()

    def init(self):  # 初始化字典
        for state in self.state_list:
            self.trans_p[state] = {}
            for state1 in self.state_list:
                self.trans_p[state][state1] = 0.0
        for state in self.state_list:
            self.initial_p[state] = 0.0
            self.emit_p[state] = {}
            self.Count_dic[state] = 0

    # 输出模型的三个参数：初始概率+转移概率+发射概率
    def output(self):
        for key in self.initial_p:  # 状态的初始概率
            self.initial_p[key] = self.initial_p[key] * 1.0 / self.line_num*1000

        for key in self.trans_p:  # 状态转移概率
            for key1 in self.trans_p[key]:
                self.trans_p[key][key1] = self.trans_p[key][key1] / self.Count_dic[key]*100

        for key in self.emit_p:  # 发射概率(状态->词语的条件概率)
            for word in self.emit_p[key]:
                self.emit_p[key][word] = self.emit_p[key][word] / self.Count_dic[key]*100

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
                        self.initial_p[line_state[0]] += 1  # initial_p记录句子第一个字的状态，用于计算初始状态概率
                        self.Count_dic[line_state[0]] += 1  # 记录每一个状态的出现次数
                    else:
                        self.trans_p[line_state[i - 1]][line_state[i]] += 1  # 用于计算转移概率
                        self.Count_dic[line_state[i]] += 1
                        if not word_list[i] in self.emit_p[line_state[i]]:
                            self.emit_p[line_state[i]][word_list[i]] = 1.0
                        else:
                            self.emit_p[line_state[i]][word_list[i]] += 1  # 用于计算发射概率
                word_list = []
                line_state = []
                continue
            else:
                word_list.append(line[0])
                line_state.append(line[2])
        self.output()
        ifp.close()

    def decode(self, sequence):
        """
        Decode the given sequence.
        """
        sequence_length = len(sequence)

        delta = {}
        for state in self.state_list:
            if not sequence[0] in self.emit_p[state]:
                delta[state] = self.initial_p[state] / self.Count_dic[state]
            else:
                delta[state] = self.initial_p[state] * self.emit_p[state][sequence[0]]

        pre = []
        for index in range(1, sequence_length):
            # if sequence[index] == "\n":
            #     continue
            delta_bar = {}
            pre_state = {}
            for state_to in self.state_list:
                max_prob = 0
                max_state = None
                for state_from in self.state_list:
                    prob = delta[state_from] * self.trans_p[state_from][state_to]
                    if prob >= max_prob:
                        max_prob = prob
                        max_state = state_from
                if not sequence[index] in self.emit_p[state_to]:
                    self.emit_p[state_to][sequence[index]] = 1.0 / self.Count_dic[state_to]
                delta_bar[state_to] = max_prob * self.emit_p[state_to][sequence[index]]
                pre_state[state_to] = max_state
            delta = delta_bar
            pre.append(pre_state)

        max_state = None
        max_prob = 0
        for state in self.state_list:
            if delta[state] >= max_prob:
                max_prob = delta[state]
                max_state = state

        if max_state is None:
            return []

        result = [max_state]
        for index in range(sequence_length - 1, 0, -1):
            max_state = pre[index - 1][max_state]
            result.insert(0, max_state)
        return ''.join(result)