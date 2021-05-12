def create_2dim_list(dim1, dim2):
    result = []
    for num1 in range(dim1):
        result.append([])
        for num2 in range(dim2):
            result[num1].append(0)
    return result


def get_content_list(url):
    fi = open(url, "r", 1, encoding='utf-8')
    content_list = []
    all_lines = fi.readlines()
    for line in all_lines:
        line = line.strip()
        for i in range(len(line)):
            content_list.append(line[i])
    return content_list


def viterbi(obs, states, temp_list, freq_dict_list):
    vtb = create_2dim_list(len(states), len(obs))
    path = create_2dim_list(len(states), len(obs))

    for i in range(len(states)):
        total_freq = 0
        for j in range(len(temp_list)):
            key_char = ""
            for k in range(len(temp_list[j][0])):
                index = temp_list[j][0][k]
                if 0 <= index < len(obs):
                    key_char += obs[index]
                else:
                    key_char += "NIL"
            if len(temp_list[j][1]) > 1:
                key_state = "NIL" + states[i]
            else:
                key_state = states[i]
            key = key_char + key_state
            total_freq += freq_dict_list[j].get(key, 0)
        vtb[i][0] = total_freq

    for i in range(1, len(obs)):
        for now_state in range(len(states)):
            max_freq = -1
            max_state = 0
            for last_state in range(len(states)):
                total_freq = 0
                for j in range(len(temp_list)):
                    key_char = ""
                    for k in range(len(temp_list[j][0])):
                        index = i + temp_list[j][0][k]
                        if 0 <= index < len(obs):
                            key_char += obs[index]
                        else:
                            key_char += "NIL"
                    if len(temp_list[j][1]) > 1:
                        key_state = states[last_state] + states[now_state]
                    else:
                        key_state = states[now_state]
                    key = key_char + key_state
                    total_freq += freq_dict_list[j].get(key, 0)
                total_freq += vtb[last_state][i - 1]
                if total_freq > max_freq:
                    max_freq = total_freq
                    max_state = last_state
            vtb[now_state][i] = max_freq
            path[now_state][i] = max_state

    max_freq = -1
    max_state = 0
    for i in range(len(states)):
        if vtb[i][len(obs) - 1] > max_freq:
            max_freq = vtb[i][len(obs) - 1]
            max_state = i
    result_int = []
    for i in range(len(obs)):
        result_int.append(0)
    result_int[len(obs) - 1] = max_state
    index = len(obs) - 2
    while index >= 0:
        result_int[index] = path[result_int[index + 1]][index + 1]
        index -= 1
    result = []
    for i in range(len(result_int)):
        result.append(states[result_int[i]])
    return result
