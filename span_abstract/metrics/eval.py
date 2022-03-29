#!/usr/bin/env python
# encoding:utf-8
# -------------------------------------------#
# Filename:
#
# Description:
# Version:       1.0
# Company:       www.10jqka.com.cn
#
# -------------------------------------------#
from pathlib import Path
import json
import os
import shutil
from span_abstract.util.interface import MetricsInterface


class EvalImpl(MetricsInterface):
    def do_eval(
        self,
        predict_file_dir: Path,
        groundtruth_file_dir: Path,
        result_json_file: Path,
        result_detail_file: Path,
        *args,
        **kargs
    ):
        """评测主函数

        Args:
            predict_file_dir (Path): input, 模型预测结果的文件目录
            groundtruth_file_dir (Path): input, 真实结果的文件目录
            result_json_file (Path): output, 评测结果，json格式，{"f1": 0.99}
            result_detail_file (Path): output, 预测明细，可选
        """
        print("Eval begin!!")
        # ==================================
        # answers, samples = read_answers(os.path.join(config.raw_data_dir(''), split + ".json"))
        # answers, samples = read_answers(file)
        labels = []
        predicts = []
        predict_num = 0
        label_num = 0
        right_num = 0
        if os.path.isfile(os.path.join(predict_file_dir, "result.txt")):
            shutil.copy(os.path.join(predict_file_dir, "result.txt"), result_detail_file)
        with open(os.path.join(groundtruth_file_dir, "span_extract_test.txt"), "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                content = line.strip().split("\t")
                if len(content) == 2:
                    g = content[0]
                    ground_span = content[1]
                    ground_spans = ground_span.split("_|_")
                else:
                    g = content[0]
                    ground_spans = []
                labels.append(ground_spans)
                label_num += len(ground_spans)
        with open(os.path.join(predict_file_dir, "result.txt"), "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                content = line.strip().split("\t")
                if len(content) == 2:
                    q = content[0]
                    predict = content[1]
                    predict_spans = predict.split("_|_")
                else:
                    g = content[0]
                    predict_spans = []
                predicts.append(predict_spans)
                predict_num += len(predict_spans)
        for i, sample in enumerate(predicts):
            for a in sample:
                try:
                    if a in labels[i]:
                        right_num += 1
                except Exception:
                    continue
        acc = float(right_num / predict_num)
        recall = float(right_num / label_num)
        f1 = float(2 * acc * recall / (acc + recall))
        results = {"acc": acc, "recall": recall, "f1": f1}
        print(results)
        with open(result_json_file, "w", encoding="utf-8") as fout:
            fout.write(json.dumps(results))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_file_dir")
    parser.add_argument("--groundtruth_file_dir")
    parser.add_argument("--result_json_file")
    parser.add_argument("--result_detail_file")
    args = parser.parse_args()
    eval_object = EvalImpl()
    eval_object.do_eval(
        args.predict_file_dir, args.groundtruth_file_dir, args.result_json_file, args.result_detail_file
    )
