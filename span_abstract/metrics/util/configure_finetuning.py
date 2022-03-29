"""Config controlling hyperparameters for fine-tuning ELECTRA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf


class FinetuningConfig(object):
    """Fine-tuning hyperparameters."""

    def __init__(self, data_dir, save_model_dir, **kwargs):
        # general
        # self.model_name = model_name
        # self.save_model_dir = save_model_dir
        self.data_dir = data_dir
        self.predict_file_dir = kwargs.get("predict_file_dir", "./")
        self.debug = False  # debug mode for quickly running things
        self.log_examples = False  # print out some train examples for debugging
        self.num_trials = 1  # how many train+eval runs to perform
        self.do_train = kwargs["do_train"]  # train a model
        self.do_eval = kwargs["do_eval"]  # evaluate the model
        self.do_test = kwargs["do_test"]  # evaluate on the test set
        self.keep_all_models = True  # if False, only keep the last trial's ckpt

        # model
        self.model_size = "small"  # one of "small", "base", or "large"
        self.task_names = kwargs["task_names"]  # which tasks to learn
        # override the default transformer hparams for the provided model size; see
        # modeling.BertConfig for the possible hparams and util.training_utils for
        # the defaults
        self.model_hparam_overrides = kwargs["model_hparam_overrides"] if "model_hparam_overrides" in kwargs else {}
        self.embedding_size = None  # bert hidden size by default
        self.vocab_size = kwargs["vocab_size"]  # number of tokens in the vocabulary
        self.do_lower_case = True

        # training
        self.learning_rate = 1e-4
        self.weight_decay_rate = 0.01
        self.layerwise_lr_decay = 0.8  # if > 0, the learning rate for a layer is
        # lr * lr_decay^(depth - max_depth) i.e.,
        # shallower layers have lower learning rates
        self.num_train_epochs = kwargs["num_train_epochs"]  # passes over the dataset during training
        self.warmup_proportion = 0.1  # how much of training to warm up the LR for
        self.save_checkpoints_steps = 1000000
        self.iterations_per_loop = 1000
        self.use_tfrecords_if_existing = True  # don't make tfrecords and write them
        # to disc if existing ones are found

        # writing model outputs to disc
        self.write_test_outputs = False  # whether to write test set outputs,
        # currently supported for GLUE + SQuAD 2.0
        self.n_writes_test = 5  # write test set predictions for the first n trials

        # sizing
        self.max_seq_length = kwargs["max_seq_length"]
        self.train_batch_size = kwargs["train_batch_size"]
        self.eval_batch_size = kwargs["eval_batch_size"]
        self.predict_batch_size = kwargs["predict_batch_size"]
        self.double_unordered = True  # for tasks like paraphrase where sentence
        # order doesn't matter, train the model on
        # on both sentence orderings for each example
        # for qa tasks
        self.max_query_length = 64  # max tokens in q as opposed to context
        self.doc_stride = 128  # stride when splitting doc into multiple examples
        self.n_best_size = 20  # number of predictions per example to save
        self.max_answer_length = 30  # filter out answers longer than this length
        self.answerable_classifier = True  # answerable classifier for SQuAD 2.0
        self.answerable_uses_start_logits = True  # more advanced answerable
        # classifier using predicted start
        self.answerable_weight = 0.5  # weight for answerability loss
        self.joint_prediction = True  # jointly predict the start and end positions
        # of the answer span
        self.beam_size = 20  # beam size when doing joint predictions
        self.qa_na_threshold = -2.75  # threshold for "no answer" when writing SQuAD
        # 2.0 test outputs

        # for multi-choice mrc
        self.evidences_top_k = 1  # top k evidences retrieved from knowledge base
        self.max_options_num = 6  # may be more than 4 with "combination" single questions
        self.answer_options = ["A", "B", "C", "D"]  # final answer options, sorted as normal
        self.max_len1 = 128  # max length of question
        self.max_len2 = 96  # max length of option
        # self.max_len3 = 288  # max length of evidence

        # TPU settings
        self.use_tpu = False
        self.num_tpu_cores = 8
        self.tpu_job_name = None
        self.tpu_name = None  # cloud TPU to use for training
        self.tpu_zone = None  # GCE zone where the Cloud TPU is located in
        self.gcp_project = None  # project name for the Cloud TPU-enabled project

        # default locations of data files
        # 数据在save_model_dir中
        self.data_dir = data_dir
        self.train_path = self.data_dir + "/train.json.out"
        self.test_path = self.data_dir + "/test.json.out"
        task_names_str = ",".join(kwargs["task_names"] if "task_names" in kwargs else self.task_names)
        self.preprocessed_data_dir = os.path.join(
            self.data_dir, "finetuning_tfrecords", task_names_str + "_tfrecords" + ("-debug" if self.debug else "")
        )
        # pretrained_model_dir = os.path.join(data_dir, "models", model_name) # 修改
        self.pretrained_model_dir = kwargs["pretrained_model_dir"]
        # self.raw_data_dir = os.path.join(data_dir, "finetuning_data", "{:}").format
        # self.raw_data_dir = os.path.join(data_dir, "{:}").format   # 修改
        self.vocab_file = os.path.join(self.pretrained_model_dir, "vocab.txt")
        # if not tf.io.gfile.exists(self.vocab_file):
        #     self.vocab_file = os.path.join(self.data_dir, "vocab.txt")

        self.init_checkpoint = self.pretrained_model_dir
        # self.save_model_dir = os.path.join(self.pretrained_model_dir, "finetuning_models",
        #                               task_names_str + "_model")
        self.save_model_dir = os.path.join(save_model_dir, "finetuning_models", task_names_str + "_model")
        results_dir = os.path.join(self.save_model_dir, "results")
        self.results_txt = os.path.join(results_dir, task_names_str + "_results.txt")
        self.results_pkl = os.path.join(results_dir, task_names_str + "_results.pkl")
        qa_topdir = os.path.join(self.predict_file_dir, task_names_str + "_qa")
        self.qa_eval_file = os.path.join(qa_topdir, "{:}_eval.json").format
        # 预测结果文件路径
        self.qa_preds_file = os.path.join(self.predict_file_dir)
        # self.qa_preds_file = os.path.join(qa_topdir, "{:}_preds.json").format
        self.pred_bad_file = None
        self.qa_na_file = os.path.join(qa_topdir, "{:}_null_odds.json").format
        # self.preprocessed_data_dir = os.path.join(
        #     self.pretrained_model_dir, "finetuning_tfrecords",
        #     task_names_str + "_tfrecords" + ("-debug" if self.debug else ""))
        self.test_predictions = os.path.join(
            self.pretrained_model_dir, "test_predictions", "{:}_{:}_{:}_predictions.pkl"
        ).format

        # default hyperparameters for single-task models
        # if len(self.task_names) == 1:
        #     task_name = self.task_names[0]
        #     if task_name == "rte" or task_name == "sts":
        #         self.num_train_epochs = 10.0
        #     elif "squad" in task_name or "qa" in task_name:
        #         self.max_seq_length = 512
        #         self.num_train_epochs = 2.0
        #         self.write_distill_outputs = False
        #         self.write_test_outputs = False
        #     elif task_name == "chunk":
        #         self.max_seq_length = 256
        #     else:
        #         self.num_train_epochs = 3.0

        # default hyperparameters for different model sizes
        if self.model_size == "large":
            self.learning_rate = 5e-5
            self.layerwise_lr_decay = 0.9
        elif self.model_size == "small":
            self.embedding_size = 128
