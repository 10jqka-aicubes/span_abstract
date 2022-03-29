#!/usr/bin/env python
# encoding:utf-8
# -----------------------------------------#
# Filename:     bm25.py.py
#
# Description:  bm25相似度检索
# Version:      1.0
# Created:      2020/6/18 14:06
# Author:       
# Company:      www.iwencai.com
#
# -----------------------------------------#

import glob
import json
import time

import dill

import jieba
import re
import heapq
import os
from gensim.summarization import bm25


class BM25Retrieval(object):
    def __init__(
        self, corpus_file_pattern=None, stop_words_file="../stop_words/stop_words.txt", MAX_LEN=300, path="./"
    ):
        """
        BM25检索模块，主要是在BM25库基础上封装了预处理部分。
        :param corpus_file_pattern: 检索资料库-文本数据 str
        :param stop_words_file: 停用词表 str
        :param path: 保存的模型目录 str
        """
        os.makedirs(path, exist_ok=True)
        self.model = os.path.join(path, "bm25.m")
        self.sen = os.path.join(path, "sen.pkl")
        self.stop = os.path.join(path, "stop.pkl")
        self.MAX_LEN = MAX_LEN
        if os.path.isfile(self.model) and os.path.isfile(self.sen) and os.path.isfile(self.stop):
            print("bm25 model found, loading...")
            self.load()
        else:
            print("training bm25 model ...")
            assert corpus_file_pattern is not None, "Can not find model or corpus file."
            if os.path.isfile(stop_words_file):
                self.stop_words = self.load_stop_words(stop_words_file)
            self.sentences, corpus = self.get_corpus(corpus_file_pattern)
            self.bm25 = bm25.BM25(corpus)
            self.dump()

    @staticmethod
    def load_stop_words(f="stop_words.txt"):
        words = set()
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                words.add(line.strip())
        return words

    def cut_and_stop(self, s):
        ws = jieba.cut(s)  # 分词
        ws = [x for x in ws if x not in self.stop_words]  # 去除停用词
        return ws

    @staticmethod
    def strQ2B(ustring):
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            # 全角区间
            if inside_code >= 0xFF01 and inside_code <= 0xFF5E:
                inside_code -= 0xFEE0
                rstring += chr(inside_code)
            # 全角空格特殊处理
            elif inside_code == 0x3000 or inside_code == 0x00A0:
                inside_code = 0x0020
                rstring += chr(inside_code)
            else:
                rstring += uchar
        return rstring

    def get_corpus(self, input_file_pattern):
        """
        句号+换行符作为段落分隔标识，连续段落不超最大长度可合并；段落超长按句号分隔。
        :param input_file_pattern:
        :return:
        """
        sentences = []
        corpus = []
        sen_tmp = ""
        for f in glob.iglob(str(input_file_pattern)):
            print(f)
            with open(f, "r", encoding="utf-8") as fp:
                lines = []
                for line in fp:
                    line = self.strQ2B(re.sub(r"\s+", "", line))  # 全角转半角
                    if len(line) < 2:
                        continue
                    lines.append(line)
                lines_str = "\n".join(lines)
                paragraphs = lines_str.split("。\n")
                for para in paragraphs:
                    if len(sen_tmp) + len(para) <= self.MAX_LEN:
                        sen_tmp += para + "。\n"
                    else:
                        words = self.cut_and_stop(sen_tmp)
                        corpus.append(words)
                        sentences.append(sen_tmp)
                        if len(para) <= self.MAX_LEN:
                            sen_tmp = para + "。\n"
                        else:
                            sen_tmp = ""
                            para_sep = para.split("。")
                            for p in para_sep:
                                if len(sen_tmp) + len(p) <= self.MAX_LEN:
                                    sen_tmp += p + "。"
                                else:
                                    words = self.cut_and_stop(sen_tmp)
                                    corpus.append(words)
                                    sentences.append(sen_tmp)
                                    sen_tmp = p + "。"
                            sen_tmp += "\n"
            if sen_tmp:
                words = self.cut_and_stop(sen_tmp)
                corpus.append(words)
                sentences.append(sen_tmp)
                sen_tmp = ""
        assert len(sentences) == len(corpus)
        print("Total paragraphs: ", len(sentences))
        return sentences, corpus

    def get_scores(self, document):
        """
        输入一个句子，返回库中所有候选的相似度
        :param document: str
        :return: List[float]
        """
        line = self.strQ2B(re.sub(r"\s+", "", document))  # 全角转半角
        tokens = self.cut_and_stop(line)
        return self.bm25.get_scores(tokens)

    def top_k(self, document, k=1):
        """
        输入document，返回最相似的k个句子。
        :param document: str
        :param k:
        :return: List[str]
        """
        scores = self.get_scores(document)
        indexes = heapq.nlargest(k, range(len(scores)), scores.__getitem__)
        return [self.sentences[i] for i in indexes]

    def dump(self):
        with open(self.model, "wb") as fpm, open(self.sen, "wb") as fpse, open(self.stop, "wb") as fpst:
            dill.dump(self.bm25, fpm)
            dill.dump(self.sentences, fpse)
            dill.dump(self.stop_words, fpst)

    def load(self):
        with open(self.model, "rb") as fpm, open(self.sen, "rb") as fpse, open(self.stop, "rb") as fpst:
            self.bm25 = dill.load(fpm)
            self.sentences = dill.load(fpse)
            self.stop_words = dill.load(fpst)


TAGS = "ABCDⅠⅡⅢⅣⅤⅥⅦⅧⅨ①②③④⑤⑥⑦⑧.、0123456789 "

# 保存bm25模型以及检索结果
def search(
    input_path, output_path, document_files, save_model_dir, mode="1", top_k=1
):  # mode=0: query only; mode=1: query+option

    datasets = input_path
    bm25_model_dir = save_model_dir + "/bm25_models/"
    bm25_retrieval = BM25Retrieval(document_files, path=bm25_model_dir)
    time_start = time.time()
    num = 0
    for f in glob.glob(str(datasets)):
        dataset_new = []
        path, filename = os.path.split(f)
        with open(f, "r", encoding="utf-8") as fp, open(output_path, "w", encoding="utf-8") as fp1:
            for line in fp:
                data = json.loads(line)
                query = data["question"]
                options = data["options"]
                evidences = {}
                if mode == "0":
                    evidence = bm25_retrieval.top_k(query.strip(TAGS), top_k)
                    for k in options.keys():
                        evidences[k] = evidence
                elif mode == "1":
                    for k, v in options.items():
                        evidence = bm25_retrieval.top_k(query.strip(TAGS) + "\n" + v.strip(TAGS), top_k)
                        evidences[k] = evidence
                data["evidences"] = evidences
                dataset_new.append(data)
            for d in dataset_new:
                fp1.write(json.dumps(d, ensure_ascii=False) + "\n")
                num += 1
    time_end = time.time()
    print("Total examples: ", num)
    print("Sec/example: ", (time_end - time_start) / num)


if __name__ == "__main__":
    pass
    # search(mode="1", top_k=1)
