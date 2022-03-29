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
import time
import shutil
import os
# from my_log import logger
import logging
import helper
from span_abstract.train.model import Model
from span_abstract.util.interface import TrainInterface


class TrainImpl(TrainInterface):
    """
    模型训练
    """

    def initialize(self, input_file_dir: Path, save_model_dir: Path, *args, **kargs):
        """训练初始化函数
        1.指定的两个参数必须使用，其他参数自定义
        Args:
            input_file_dir (Path): input, 训练文件的目录
            save_model_dir (Path): output, 保存模型的目录
        """
        self.input_file_dir = input_file_dir
        self.save_model_dir = save_model_dir
        self.bert_model_dir = kargs["bert_model_dir"]
        self.dev_file_dir = kargs["dev_file_dir"]
        self.task_name = "ner"
        print("Train init!!")

    def do_train(self):
        """
        训练主函数
        """
        print("Train begin!!")
        para = {"lstm_dim": 128, "max_epoch": 20, "train_batch": 32, "dev_batch": 32, "require_improvement": 1000}

        logging.warn("--------" * 10)
        logging.warn("\npara : \n {para}".format(para=json.dumps(para, indent=4, ensure_ascii=False)))
        base_config = {
            "task_name": "wencai_span",
            "mode": "bert",
            "lstm_dim": 128,
            "embedding_size": 50,
            "max_epoch": 10,
            "train_batch": 16,
            "dev_batch": 16,
            "learning_rate": 1e-4,
            "require_improvement": 1000,
            "bert_config": os.path.join(self.bert_model_dir, "bert_config_rbt3.json"),
            "init_checkpoint": os.path.join(self.bert_model_dir, "bert_model.ckpt"),
            "vocab_dir": os.path.join(self.bert_model_dir, "vocab.txt"),
            "checkpoint_dir": "./result/{task_name}/ckpt_model/{model_version}".format(
                task_name=self.task_name, model_version=time.strftime("%Y%m%d")
            ),  # %Y%m%d%H%M%S
            "checkpoint_path": "./result/{task_name}/ckpt_model/{model_version}/{task_name}.ckpt".format(
                task_name=self.task_name, model_version=time.strftime("%Y%m%d")
            ),
            "train_file": os.path.join(self.input_file_dir, "span_extract_train.txt"),
            "dev_file": os.path.join(self.input_file_dir, "span_extract_dev.txt"),
            "predict_file": "data/{task_name}/predict".format(task_name=self.task_name),
            "predict_result": "data/{task_name}/predict_result".format(task_name=self.task_name),
            "tf_serving_save_dir": os.path.join(self.save_model_dir, self.task_name),
            "parameter_information": os.path.join(self.save_model_dir, "parameter_information.json"),
            "save_dir": os.path.join(self.save_model_dir),
            "tag_to_id": os.path.join(self.save_model_dir, "tag_to_id.json"),
            "id_to_tag": os.path.join(self.save_model_dir, "id_to_tag.json"),
        }

        bert_config = helper.obj_load(base_config["bert_config"])
        base_config = helper.merge_two_dicts(base_config, para)
        config = {"base_config": base_config, "bert_config": bert_config}
        helper.obj_save(config, base_config["parameter_information"])
        if os.path.exists(os.path.join(base_config["save_dir"], "vocab.txt")):
            logging.debug(["model_result vocab_file existed!!"])
        else:
            shutil.copy(base_config["vocab_dir"], base_config["save_dir"])
        logging.info(base_config)

        model = Model(base_config)

        model.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_dir")
    parser.add_argument("--save_model_dir")
    parser.add_argument("--bert_model_dir")
    parser.add_argument("--dev_file_dir")
    args = parser.parse_args()
    train_object = TrainImpl(
        args.input_file_dir,
        args.save_model_dir,
        bert_model_dir=args.bert_model_dir,
        dev_file_dir=args.dev_file_dir
    )
    train_object.do_train()
