class ExampleModel:
    def predict(self, sentence):
        samples = open("example_dataset/input.utf8", encoding="utf8")
        samples = [ele.strip() for ele in samples]
        outputs = open("example_dataset/output.utf8", encoding="utf8")
        outputs = [ele.strip() for ele in outputs]
        if sentence == "我爱北京天安门":
            return "SSBEBIE"
        elif sentence == "今天天气怎么样":
            return "BEBEBIE"
        else:
            if sentence in samples:
                return outputs[samples.index(sentence)]

