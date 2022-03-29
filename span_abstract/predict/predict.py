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
from span_abstract.util.interface import PredictInterface
import os
import argparse
from model_client_server import SpanPrediction


class PredictImpl(PredictInterface):
    def initialize(self, input_file_dir: Path, load_model_dir: Path, predict_file_dir: Path, *args, **kargs):
        """预测初始化函数
        1.指定的三个参数必须使用，其他参数自定义
        Args:
            input_file_dir (Path): input, 预测的文件目录
            load_model_dir (Path): input, 加载模型的目录
            predict_file_dir (Path): output, 预测结果的文件目录
        """
        self.input_file_dir = input_file_dir
        self.load_model_dir = load_model_dir
        self.predict_file_dir = predict_file_dir
        print("Predict init!!")
        # processors = data_processor.NerProcessor
        # base_config = helper.obj_load("../model/saved_model/parameter_information.json")["base_config"]
        self.model = SpanPrediction(
            load_model_dir=self.load_model_dir
        )
        self.model._config_()

    def do_predict(self):
        """
        预测主函数
        """
        print("Predict begin!!")
        # ==================================
        # write your code
        # ==================================
        reses = []
        with open(os.path.join(self.input_file_dir, "span_extract_test.txt"), "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                content = line.strip().split("\t")
                query = content[0]
                result = self.model.predict(query)
                # print(result)
                r = []
                for e in result:
                    r.append(e["words"])
                reses.append([query, "_|_".join(r)])
        with open(os.path.join(self.predict_file_dir, "result.txt"), "w") as f2:
            f2.write("用户问句\t名词短语\n")
            for r in reses:
                f2.write(r[0] + "\t" + r[1] + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_dir")
    parser.add_argument("--load_model_dir")
    parser.add_argument("--predict_file_dir")
    args = parser.parse_args()
    predict_object = PredictImpl(args.input_file_dir, args.load_model_dir, args.predict_file_dir)
    predict_object.do_predict()
