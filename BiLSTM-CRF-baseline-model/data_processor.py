import json
from vocabulary import Vocabulary

class CluenerProcessor:
    """Processor for the chinese ner data set."""
    def __init__(self,data_dir):
        self.vocab = Vocabulary()
        self.data_dir = data_dir

    def get_vocab(self):
        vocab_path = self.data_dir / 'vocab.pkl'
        if vocab_path.exists():
            self.vocab.load_from_file(str(vocab_path))
        else:
            files = ["train.txt", "dev.txt", "test.txt"]
            for file in files:
                with open(str(self.data_dir / file), 'r') as fr:
                    for line in fr:
                        text = line.strip().split(" ")[0]
                        self.vocab.update(list(text))
            self.vocab.build_vocab()
            self.vocab.save(vocab_path)

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "train.txt"), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "dev.txt"), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "test.txt"), "test")

    def _create_examples1(self,input_path,mode):
        examples = []
        with open(input_path, 'r') as f:
            idx = 0
            for line in f:
                json_d = {}
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label', None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                json_d['id'] = f"{mode}_{idx}"
                json_d['context'] = " ".join(words)
                json_d['tag'] = " ".join(labels)
                json_d['raw_context'] = "".join(words)
                idx += 1
                examples.append(json_d)
        return examples

    # 读取数据集
    def _create_examples(self,input_path,mode):
        # 读取数据集
        with open(input_path, "r", encoding="utf-8") as f:
            content = [_.strip() for _ in f.readlines()]

        # 添加原文句子以及该句子的标签
        # 读取空行所在的行号
        index = [-1]
        index.extend([i for i, _ in enumerate(content) if ' ' not in _])
        index.append(len(content))

        # 按空行分割，读取原文句子及标注序列
        sentences, tags = [], []
        examples = []
        idx = 0
        for j in range(len(index)-1):
            json_d = {}
            sent, tag = [], []
            segment = content[index[j]+1: index[j+1]]
            for line in segment:
                sent.append(line.strip().split(" ")[0])
                tag.append(line.strip().split(" ")[-1])

            sentences.append(' '.join(sent))
            tags.append(tag)

            json_d['id'] = f"{mode}_{idx}"
            json_d['context'] = " ".join(sent)
            json_d['tag'] = " ".join(tag)
            json_d['raw_context'] = "".join(sent)
            idx += 1
            examples.append(json_d)

        return examples



