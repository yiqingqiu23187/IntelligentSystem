from typing import List
import torch
import pickle


class Solution:
    # --------------------
    #  学号 18302010034 和 用户名 黄子豪
    # --------------------
    ID = "18302010034"
    NAME = "Huangzihao"

    # --------------------
    # 对于下方的预测接口，需要实现对你模型的调用：
    #
    # 要求：
    #    输入：一组句子
    #    输出：一组预测标签
    #
    # 例如：
    #    输入： ["我爱北京天安门", "今天天气怎么样"]
    #    输出： ["SSBEBIE", "BEBEBIE"]
    # --------------------

    # --------------------
    # 一个样例模型的预测
    # --------------------
    def example_predict(self, sentences: List[str]) -> List[str]:
        # from .example_model import ExampleModel
        #
        # model = ExampleModel()
        # results = []
        # for sent in sentences:
        #     results.append(model.predict(sent))
        # return results
        pass

    # --------------------
    # HMM 模型的预测接口
    # --------------------
    def hmm_predict(self, sentences: List[str]) -> List[str]:
        from wordseg.HmmModel import HmmModel
        model = HmmModel("../dataset2/train.utf8")
        model.train()
        results = []
        for sent in sentences:
            # results.append(model.predict(sent))
            if not sent:
                #results.append('')
                continue
            results.append(model.decode(sent))
        return results
        #pass

    # --------------------
    # CRF 模型的预测接口
    # --------------------
    def crf_predict(self, sentences: List[str]) -> List[str]:
        from wordseg import Simple_Func
        import pickle
        with open('./crf_model/model50.pkl', 'rb') as fr:
            temp_list = pickle.load(fr)
            freq_dict_list = pickle.load(fr)
        results = []
        for sent in sentences:
            results.append("".join(Simple_Func.viterbi(sent, "BEIS", temp_list, freq_dict_list)))
        return results
        #pass

    # --------------------
    # DNN 模型的预测接口
    # --------------------
    def dnn_predict(self, sentences: List[str]) -> List[str]:
       # model = torch.load('LSTMmodel/lstmmodel0.pkl')
        model = torch.load('LSTMmodel/model3.pkl')
        with open('LSTMmodel/datasave.pkl', 'rb') as inp:
            word2id = pickle.load(inp)
            id2word = pickle.load(inp)
            tag2id = pickle.load(inp)
            id2tag = pickle.load(inp)
        inp.close()
        length = len(id2word)
        result = []
        for sentence in sentences:
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
            result.append(''.join(y))

        return result
       #from wordseg import LSTM
       # model = torch.load('LSTMmodel/model3.pkl')
       # results = []
       # for sent in sentences:
       #     score, tag_seq = model.test(sent)
       #     results.append(''.join(tag_seq))
       # return results
