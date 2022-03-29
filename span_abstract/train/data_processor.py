# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/10 4:46 PM
# @Author: wuchenglong


import copy
import helper
import random


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, tokenizer, entry, config):
        self.tokenizer = tokenizer
        self.data = []
        self.input_size = 0
        self.vocab = {}
        self.batch_data = []
        self.tag_to_id_dir = config["tag_to_id"]
        self.id_to_tag_dir = config["id_to_tag"]
        self.entry = entry
        if entry == "train":
            self.data_path = config["train_file"]
            self.batch_size = config["train_batch"]
        elif entry == "dev":
            self.data_path = config["dev_file"]
            self.batch_size = config["dev_batch"]

        self.build_tag()
        self.load_data()
        self.prepare_batch()

    def get_tags(self):
        raise NotImplementedError()

    def build_tag(self):
        self.tag_to_id = dict(zip(self.get_tags(), range(len(self.get_tags()))))
        self.id_to_tag = dict(zip(range(len(self.get_tags())), self.get_tags()))
        helper.obj_save(self.tag_to_id, self.tag_to_id_dir)
        helper.obj_save(self.id_to_tag, self.id_to_tag_dir)

    def sep_tag(self):
        raise NotImplementedError()

    def transform(self, line):
        # 转化数据格式
        content = line.strip().split("\t")
        try:
            query = content[0]
            spans = content[1]
            pos = []
            span_s = spans.split("_|_")
            for c in span_s:
                if len(c) == 0:
                    continue
                q = query.lower()
                # print (c.lower(),q)
                try:
                    # pos = [[substr.start(),substr.end()] for substr in re.finditer(re.escape(c.lower()),q)]
                    begin_pos = q.index(c.lower())
                    # for begin_pos in begin_pos_lists:
                    end_pos = begin_pos + len(c)
                    pos.append([begin_pos, end_pos])
                except Exception as e:
                    print("错误信息", e)
                    continue
        except Exception:
            print(line)
            query = content[0]
            pos = []

        q = list(query)
        key = "NOUN"
        tag = ["0"] * len(q)
        for p in pos:
            if len(p) != 2:
                continue
            # print (len(pos))
            begin = p[0]
            end = p[1]
            tag[begin] = "B-" + key
            tag[end - 1] = "E-" + key
            for _p in range(begin + 1, end - 1):
                tag[_p] = "I-" + key
        return q, tag

    def load_data(self):
        tokens = ["[CLS]"]
        target = ["[CLS]"]
        train_nums = 0
        with open(self.data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                line = line.rstrip()
                train_nums += 1
                words, tags = self.transform(line)
                # print (words)
                # print (tags)
                for i, word in enumerate(words):
                    tag = tags[i]
                    # word, tag = None,None
                    # try:
                    #     word, tag = line.split()
                    # except Exception as error:
                    if word and tag:
                        tokens.append(word.lower())
                        target.append(tag)
                tokens.append("[SEP]")
                target.append("[SEP]")
                # import pdb;pdb.set_trace()
                inputs_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                segment_ids = [0] * len(inputs_ids)
                input_mask = [1] * len(inputs_ids)
                tag_ids = self.convert_tag(target)

                data = [tokens, tag_ids, inputs_ids, segment_ids, input_mask]
                # num = random.randint(1,5)
                # if num > 2:
                #     continue
                self.data.append(data)

                tokens = ["[CLS]"]
                target = ["[CLS]"]

        random.shuffle(self.data)

    def convert_tag(self, tag_list):
        return [self.tag_to_id[tag] for tag in tag_list]

    def prepare_batch(self):
        """
        prepare data for batch
        """
        index = 0
        while True:
            if index + self.batch_size > len(self.data):
                pad_data = self.pad_data(self.data[-self.batch_size :])
                self.batch_data.append(pad_data)
                break
            else:
                pad_data = self.pad_data(self.data[index : index + self.batch_size])
                index += self.batch_size
                self.batch_data.append(pad_data)

    def pad_data(self, data):
        c_data = copy.deepcopy(data)
        max_length = max([len(i[0]) for i in c_data])
        padded_data = []
        for i in c_data:
            tokens, tag_ids, inputs_ids, segment_ids, input_mask = i
            tag_ids = tag_ids + (max_length - len(tag_ids)) * [0]
            inputs_ids = inputs_ids + (max_length - len(inputs_ids)) * [0]
            segment_ids = segment_ids + (max_length - len(segment_ids)) * [0]
            input_mask = input_mask + (max_length - len(input_mask)) * [0]
            assert len(tag_ids) == len(inputs_ids) == len(segment_ids) == len(input_mask)
            padded_data.append([tokens, tag_ids, inputs_ids, segment_ids, input_mask])
        return padded_data

    def iteration(self):
        idx = 0
        while True:
            yield self.batch_data[idx]
            idx += 1
            if idx > len(self.batch_data) - 1:
                idx = 0

    def get_batch(self):
        for data in self.batch_data:
            yield data


class ExampleProcessor(DataProcessor):
    def get_tags(self):
        return ["-", "[SEP]", "[CLS]", "O"]

    def sep_tag(self):
        return "end"


class NerProcessor(DataProcessor):
    def get_tags(self):
        return ["-", "[SEP]", "[CLS]", "0", "B-NOUN", "I-NOUN", "E-NOUN"]

    def sep_tag(self):
        return "end"


if __name__ == "__main__":
    import tokenization

    tokenizer = tokenization.FullTokenizer(
        vocab_file="data/vocab.txt",
    )
