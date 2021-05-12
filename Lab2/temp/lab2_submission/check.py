from wordseg.solution import Solution


print("\n\n\n*************************************")
print(f"检查姓名学号 >>>")
assert Solution.ID != "", "请输入学号！"
assert Solution.NAME != "", "请输入姓名！"
print("[√]通过检查")


solution = Solution()
inputs = ["我爱北京天安门", "今天天气怎么样"]

funcs = {
    "EXAMPLE": solution.example_predict,
    "HMM": solution.hmm_predict,
    "CRF": solution.crf_predict,
    "DNN": solution.dnn_predict,
}

print("\n\n\n*************************************")
print(f"检查输入输出格式 >>>")
for f_name, func in funcs.items():
    print(f"\n检查{f_name}模型 -->")
    outputs = func(inputs)
    if outputs == None:
        print(f"{f_name}模型没有输出。")
        print("\t[×]未通过检查")
        continue
    print(f"对于输入{inputs}，你的输出是{outputs}。")
    if not isinstance(outputs, list):
        print(f"\t{f_name}模型的输出格式有问题：应该输出列表。")
        print("[×]未通过检查")
        continue
    if not isinstance(outputs[0], str):
        print(f"\t{f_name}模型的输出格式有问题：列表元素应该为字符串。")
        print("[×]未通过检查")
        continue
    if len(outputs[0]) != len(inputs[0]) or len(outputs[1]) != len(inputs[1]):
        print(f"\t{f_name}模型的输出和输入不等长，请检查。")
        print("[×]未通过检查")
        continue
    print(f"[√] 通过检查")

print("\n\n\n*************************************")
print(f"检查样例数据集 >>>")

for f_name, func in funcs.items():
    print(f"\n检查{f_name}模型 -->")
    examples = open("example_dataset/input3.utf8", encoding="utf8").readlines()
    examples = [ele.strip() for ele in examples]
    gold = open("example_dataset/gold3.utf8", encoding="utf8").readlines()
    gold = [ele.strip() for ele in gold]
    pred = func(examples)
    accuracys = []
    if pred is not None:
        for i in range(len(examples)):
            # print(f"\n输入：{examples[i]}")
            # print(f"输出：{pred[i]}")
            # print(f"正确：{gold[i]}")
            corr = [1 if a == b else 0 for a, b in zip(gold[i], pred[i])]
            accu = sum(corr) / len(corr)
            accuracys.append(accu)
            # print(f"准确率：{accu}")
        print(f"准确率：{sum(accuracys)/len(accuracys)}")
    else:
        print(f"{f_name}暂无结果")
