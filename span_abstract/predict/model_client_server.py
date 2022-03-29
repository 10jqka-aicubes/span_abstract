# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/10 4:46 PM
# @Author: wuchenglong
import os
import tensorflow as tf
import tensorflow.keras as kr
import tokenization
import helper


class Prediction(object):
    def __init__(self):
        pass

    def _config_(self, model):
        """模型调用的一些配置信息"""
        pass

    def input_process(self):
        """对输入信息进行预处理"""
        pass

    def predict(self, text):
        """调用模型预测"""
        pass

    def path_id_to_tag(self, path):
        """对路径转换成tag"""
        pass

    def result(self, text):
        """对模型结果机构化，返回预测结果"""
        pass

    def _format_result(self, chars, path):
        """一些模型结构化的处理"""
        pass


_CPU_NUM = 4


class SpanPrediction(Prediction):

    def __init__(self, load_model_dir):
        self.task_name = "ner"
        self.load_model_dir = load_model_dir

    def _config_(
        self,
    ):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(self.load_model_dir, "vocab.txt"))
        # self.model_name = model
        self.time_out = 10
        self.id_to_tag = helper.obj_load(os.path.join(self.load_model_dir, "id_to_tag.json"))
        self.construct_param(os.path.join(self.load_model_dir, self.task_name))
        print(self.id_to_tag)

    def update_input(self, text_list):
        self.text_list = text_list

    def input_process(self):
        input_ids = []
        input_mask = []
        segment_ids = []

        for text in self.text_list:
            tokens = ["[CLS]"] + list(text) + ["[SEP]"]
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))
            input_mask.append([1] * len(tokens))
            segment_ids.append([0] * len(tokens))

        self.max_length = max([len(elem) for elem in input_ids])

        # 使用pad_sequences来将文本pad为固定长度
        input_ids = kr.preprocessing.sequence.pad_sequences(
            input_ids, self.max_length, padding="post", truncating="post"
        )
        input_mask = kr.preprocessing.sequence.pad_sequences(
            input_mask, self.max_length, padding="post", truncating="post"
        )
        segment_ids = kr.preprocessing.sequence.pad_sequences(
            segment_ids, self.max_length, padding="post", truncating="post"
        )

        input = {"input_ids": input_ids, "segment_ids": segment_ids, "input_mask": input_mask, "dropout": 1}
        return input

    def path_id_to_tag(self, path):
        return [self.id_to_tag[str(elem)] for elem in path]

    def construct_param(self, model_path):
        pb_model_path = model_path
        cpu_num = _CPU_NUM
        tf_config = tf.ConfigProto(
            device_count={"CPU": cpu_num},
            inter_op_parallelism_threads=cpu_num,
            intra_op_parallelism_threads=cpu_num,
            allow_soft_placement=True,
        )
        # tf_config.gpu_options.allow_growth = True
        # if not os.path.isfile(os.path.join(_MODEL_PATH,'wencai.pb')):
        sess = tf.Session(graph=tf.Graph(), config=tf_config)
        tf.saved_model.loader.load(sess, ["serve"], pb_model_path)
        graph = sess.graph
        # constant_graph = tf.graph_util.convert_variables_to_constants(
        #     sess, sess.graph_def, ["loss_layer/outputs", "logits/Reshape", "Sum/reduction_indices"]
        # )
        #     with gfile.FastGFile(os.path.join(_MODEL_PATH,'wencai.pb'), 'wb') as f:
        #         f.write(constant_graph.SerializeToString())
        # else:
        #     sess = tf.Session(config=tf_config)
        #     with gfile.FastGFile(os.path.join(_MODEL_PATH,'wencai.pb'), 'rb') as f:
        #         graph_def = tf.GraphDef()
        #         graph_def.ParseFromString(f.read())
        #         sess.graph.as_default()
        #         tf.import_graph_def(graph_def, name='')
        graph = sess.graph
        self.sess = sess
        # graph = tf.get_default_graph()
        # graph = self.sess.graph
        # input_graph_def = graph.as_graph_def()
        # with open('node', 'w') as f2:
        #     for n in input_graph_def.node:
        #         f2.write(n.name + '\n')
        self.input_ids = graph.get_tensor_by_name("input_ids:0")
        self.input_mask = graph.get_tensor_by_name("input_mask:0")
        self.segment_ids = graph.get_tensor_by_name("segment_ids:0")
        self.dropout = graph.get_tensor_by_name("dropout:0")
        self.logits = graph.get_tensor_by_name("logits/Reshape:0")
        self.length = graph.get_tensor_by_name("Sum/reduction_indices:0")
        self.pre_paths = graph.get_tensor_by_name("loss_layer/outputs:0")

    def prepare_pred_data(self, text):
        max_length = len(text) + 2
        tokens = list(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        # logger.info(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        input_ids = input_ids + (max_length - len(input_ids)) * [0]
        segment_ids = segment_ids + (max_length - len(segment_ids)) * [0]
        input_mask = input_mask + (max_length - len(input_mask)) * [0]

        feed = {
            self.input_ids: [input_ids],
            self.segment_ids: [segment_ids],
            self.input_mask: [input_mask],
            self.dropout: 1.0,
        }
        return feed

    def predict(self, text):
        # input = self.input_process()
        # request.model_spec.name = self.model_name
        # # request.model_spec.version.value = 20191113
        # request.model_spec.signature_name = self.signature_name
        # request.inputs["input_ids"].CopyFrom(
        #     tf.contrib.util.make_tensor_proto(input["input_ids"], shape=[len(input["input_ids"]), self.max_length]))
        # request.inputs["segment_ids"].CopyFrom(
        #     tf.contrib.util.make_tensor_proto(input["segment_ids"], shape=[len(input["input_ids"]), self.max_length]))
        # request.inputs["input_mask"].CopyFrom(
        #     tf.contrib.util.make_tensor_proto(input["input_mask"], shape=[len(input["input_ids"]), self.max_length]))
        # request.inputs['dropout'].CopyFrom(
        #     tf.contrib.util.make_tensor_proto(input["dropout"])
        print("predict...")
        feed = self.prepare_pred_data(text)
        # print ('feed:',feed)
        # feed_dict = {
        #               self.input_ids: input["input_ids"],
        #               self.input_mask: input["input_mask"],
        #               self.segment_ids: input["segment_ids"],
        #               self.dropout:input["dropout"]
        #             }
        # paths = self.sess.run([self.pre_paths], feed_dict=feed)
        logits, length, paths = self.sess.run([self.logits, self.length, self.pre_paths], feed_dict=feed)
        # print(response.model_spec.version.value)
        # result = stub.Predict(request, time_out).outputs["pre_paths"].int_val  # 10 secs timeout
        # pre_path_list = np.array(response.outputs["pre_paths"].int_val).reshape((-1, self.max_length))
        # paths = paths.tolist()
        # print("paths:", paths)
        entities_result = helper.format_result(
            ["[CLS]"] + list(text) + ["[SEP]"], [self.id_to_tag[str(elem)] for elem in paths[0]]
        )
        return entities_result

    def result(self, text):
        pre_path_list = self.predict(text)
        text_list = [["[CLS]"] + list(elem) + ["[SEP]"] for elem in self.text_list]
        result_list = []
        for text, path in list(zip(text_list, pre_path_list)):
            result_list.append(self._format_result(text, path))
        return result_list

    def _format_result(self, chars, path):
        tags = self.path_id_to_tag(path)
        return helper.format_result(chars, tags)


if __name__ == "__main__":
    pass