from sklearn.model_selection import train_test_split
import pickle

INPUT_DATA = "../../dataset1/train.utf8"
SAVE_PATH = "../LSTMmodel/datasave.pkl"
id2tag = ['B', 'I', 'E', 'S']  # B：分词头部 I：分词词中 E：分词词尾 S：独立成词
tag2id = {'B': 0, 'I': 1, 'E': 2, 'S': 3}
word2id = {}
id2word = []


# def handle_data():
#     x_data=[]
#     y_data=[]
#     wordnum=0
#     line_num=0
#     with open(INPUT_DATA,'r',encoding="utf-8") as ifp:
#         for line in ifp:
#             if not line or line in ['\n','\r\n']:
#                 continue
#             line_num = line_num+1
#             line = line.strip()
#             line_x = []
#             line = line.split(" ")
#
#             if(line[0] in id2word):
#                 line_x.append(word2id[line[0]])
#             else:
#                 id2word.append(line[0])
#                 word2id[line[0]]=wordnum
#                 line_x.append(wordnum)
#                 wordnum=wordnum+1
#             x_data.append(line_x)
#             y_data.append(tag2id[line[1]])
#
#     x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=43)
#     with open(SAVE_PATH, 'wb') as outp:
#         pickle.dump(word2id, outp)
#         pickle.dump(id2word, outp)
#         pickle.dump(tag2id, outp)
#         pickle.dump(id2tag, outp)
#         pickle.dump(x_train, outp)
#         pickle.dump(y_train, outp)
#         pickle.dump(x_test, outp)
#         pickle.dump(y_test, outp)
def handle_data():
    x_data = []  # 所有的字(下标)
    y_data = []  # 所有的标签
    wordnum = 0
    line_num = 0

    with open(INPUT_DATA, 'r', encoding="utf-8") as ifp:
        line_x = []
        line_y = []
        for line in ifp:
            line = line.strip()
            if not line or line in ['\n', '\r\n']:
                line_num += 1
                x_data.append(line_x)
                y_data.append(line_y)
                line_x = []
                line_y = []
                continue
            else:
                if (line[0] in id2word):
                    line_x.append(word2id[line[0]])
                else:
                    id2word.append(line[0])
                    word2id[line[0]] = wordnum
                    line_x.append(wordnum)
                    wordnum = wordnum + 1
                line_y.append(tag2id[line[2]])
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.01, random_state=43)

    with open(SAVE_PATH, 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)


if __name__ == "__main__":
    handle_data()
