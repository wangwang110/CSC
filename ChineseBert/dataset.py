from torch.utils.data import DataLoader, Dataset


class BertDataset(Dataset):
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        data = self.dataset[index]
        return data


def construct(filename):
    f = open(filename, encoding='utf8')
    list = []
    for line in f:
        try:
            line = line.replace("\n", "")
            pairs = line.split(" ")
            # print(pairs)
            elem = {'input': pairs[0], 'output': pairs[1]}
            list.append(elem)
        except Exception as e:
            print(e)
            continue
    return list


def construct_pretrain(filename):
    f = open(filename, encoding='utf8')
    list = []
    for line in f:
        try:
            line = line.replace("\n", "")
            pairs = line.split(" ")
            elem = {'input': "", 'output': pairs[0]}
            list.append(elem)
        except Exception as e:
            print(e)
            continue
    return list


def construct_ner(filename):
    f = open(filename, encoding='utf8')
    ner_li = open("/data_local/TwoWaysToImproveCSC/BERT/data/merge_train_ner_tag.txt", encoding='utf8').readlines()

    list = []
    i = 0
    for line in f:
        line = line.replace("\n", "")
        pairs = line.split(" ")
        ner_ids_str = ner_li[i]
        # print(ner_ids_str)
        try:
            elem = {'input': pairs[0], 'output': pairs[1], 'output_ner': ner_ids_str}
            list.append(elem)
        except Exception as e:
            print(e)
            continue
        i += 1
    return list


def singleconstruct(filename):
    f = open(filename, encoding='utf8')
    list = []
    for line in f:
        line = line.replace("\n", "")
        pairs = line.split(" ")
        if (len(pairs[0]) != len(pairs[1])):
            continue
        elem = {'input': pairs[1], 'output': pairs[1]}
        list.append(elem)
    return list


def testconstruct(filename):
    f = open(filename, encoding='utf8')
    list = []
    for line in f:
        line = line.replace("\n", "")
        pairs = line.split(" ")
        elem = {'input': pairs[0], 'output': ""}
        list.append(elem)
    return list


def cc_testconstruct(filename):
    f = open(filename, encoding='utf8')
    list = []
    for line in f:
        line = line.replace("\n", "")
        pairs = line.split(" ")
        elem = {'output': pairs[0], 'input': pairs[1]}
        list.append(elem)
    return list


def li_testconstruct(sent_li):
    list = []
    for line in sent_li:
        line = line.replace("\n", "")
        pairs = line.split(" ")
        elem = {'input': pairs[0], 'output': ""}
        list.append(elem)
    return list
