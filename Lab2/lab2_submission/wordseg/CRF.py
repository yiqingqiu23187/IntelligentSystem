from wordseg import Simple_Func
import pickle


# 特征模板定义
tempCS0 = [[-1], [0]]
tempC0S0 = [[0], [0]]
tempC1S0 = [[1], [0]]
tempCC0S0 = [[-1, 0], [0]]
tempC0C1S0 = [[0, 1], [0]]
tempCC1S0 = [[-1, 1], [0]]

tempCSS0 = [[-1], [-1, 0]]
tempC0SS0 = [[0], [-1, 0]]
tempC1SS0 = [[1], [-1, 0]]
tempCC0SS0 = [[-1, 0], [-1, 0]]
tempC0C1SS0 = [[0, 1], [-1, 0]]
tempCC1SS0 = [[-1, 1], [-1, 0]]
temp_list = []
temp_list.append(tempCS0)
temp_list.append(tempC0S0)
temp_list.append(tempC1S0)
temp_list.append(tempCC0S0)
temp_list.append(tempC0C1S0)
temp_list.append(tempCC1S0)

temp_list.append(tempCSS0)
temp_list.append(tempC0SS0)
temp_list.append(tempC1SS0)
temp_list.append(tempCC0SS0)
temp_list.append(tempC0C1SS0)
temp_list.append(tempCC1SS0)

freq_dict_list = []
for i in range(len(temp_list)):
    freq_dict = {}
    freq_dict_list.append(freq_dict)

# 读入训练集
fi = open("../../dataset2/train.utf8", "r", 1, encoding='utf-8')
all_lines = fi.readlines()
all_char = ""
all_state = ""
for line in all_lines:
    line = line.strip()
    if line != "":
        all_char = all_char + line[0]
        all_state = all_state + line[2]
fi.close()



for i in range(len(all_char)):
    for j in range(len(temp_list)):
        key_char = ""
        for k in range(len(temp_list[j][0])):
            index = i + temp_list[j][0][k]
            if 0 <= index < len(all_char):
                key_char += all_char[index]
            else:
                key_char += "NIL"
        key_state = ""
        for k in range(len(temp_list[j][1])):
            index = i + temp_list[j][1][k]
            if 0 <= index < len(all_char):
                key_state += all_state[index]
            else:
                key_state += "NIL"
        key = key_char + key_state
        freq_dict_list[j][key] = freq_dict_list[j].get(key, 0) + 1



count = 1
while True:
    print("第" + str(count) + "轮")
    count += 1
    hit = len(all_state)
    # result1 = viterbi(all_char, "BEIS")
    result1 = Simple_Func.viterbi(all_char, "BEIS", temp_list, freq_dict_list)
    for i in range(len(all_state)):
        if result1[i] != all_state[i]:
            hit -= 1
            for j in range(len(temp_list)):
                key_char = ""
                for k in range(len(temp_list[j][0])):
                    index = i + temp_list[j][0][k]
                    if 0 <= index < len(all_char):
                        key_char += all_char[index]
                    else:
                        key_char += "NIL"
                key_state = ""
                for k in range(len(temp_list[j][1])):
                    index = i + temp_list[j][1][k]
                    if 0 <= index < len(all_char):
                        key_state += all_state[index]
                    else:
                        key_state += "NIL"
                key1 = key_char + key_state
                freq_dict_list[j][key1] = freq_dict_list[j].get(key1, 0) + 1

                char1 = key_char
                state1 = result1[i]
                if len(temp_list[j][1]) > 1:
                    if i - 1 < 0:
                        state1 = "NIL" + result1[i]
                    else:
                        state1 = result1[i - 1] + result1[i]
                key1 = char1 + state1
                freq_dict_list[j][key1] = freq_dict_list[j].get(key1, 0) - 1
    print(hit / len(all_state))

    obs_test = Simple_Func.get_content_list("../example_dataset/input.utf8")
    gold_test = Simple_Func.get_content_list("../example_dataset/gold.utf8")
    result_test = Simple_Func.viterbi(obs_test, "BEIS", temp_list, freq_dict_list)
    hit = 0
    for i in range(len(result_test)):
        # print(obs_test[i] + " " + gold_test[i] + " " + result_test[i])
        if result_test[i] == gold_test[i]:
            hit += 1
    print(hit / len(result_test))

    with open("../crf_model/model" + str(count) + ".pkl", 'wb') as outp:  # 保存
        pickle.dump(temp_list, outp)
        pickle.dump(freq_dict_list, outp)
