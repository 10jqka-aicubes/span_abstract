# 【2-4春季赛】名词短语抽取算法任务

​	名词短语抽取是NLP的一个基础抽取任务，它相较于名词的优势在于，名词短语能提供更多的有效信息来表达文本中心含义。精准的名词短语能提升各项下游任务的效果，如知识图谱的构建、对话系统的应用等等。在真实业务场景下，名词短语抽取常常面临着语料口语化、新知识、一词多义、修饰词组合等挑战。

- 本代码是该赛题的一个基础demo，仅供参考学习。
- 比赛地址：http://contest.aicubes.cn/	
- 时间：2022-02 ~ 2022-04



## 如何运行Demo

- clone代码


- 下载预训练模型，存放在参数`bert_model_dir`对应路径下

  ```
  https://github.com/ymcui/Chinese-BERT-wwm  -> RBT3, Chinese Tensorflow
  ```

- 准备环境

  - cuda10.0以上

  - python3.7以上

  - 安装python依赖

    ```
    python -m pip install -r requirements.txt
    ```

- 准备数据，，从[官网](http://contest.aicubes.cn/#/detail?topicId=48)下载数据

  - 从`train/span_extract_train.txt`切出一部分作为验证集，命名为`span_extract_dev.txt`，训练集和验证集分别存放在`--input_file_dir`和`--dev_file_dir`对应的路径下


- 调整参数配置，参考[模板项目](https://github.com/10jqka-aicubes/project-demo)的说明，主要配置文件`span_abstract/setting.conf`


- 运行

  - 训练

  ```
  bash span_abstract/train/run.sh
  ```

  - 预测

  ```
  bash span_abstract/predict/run.sh
  ```

  - 计算结果指标

  ```
  bash span_abstract/metrics/run.sh
  ```



## 提交B榜代码规范

- 参考[模板项目](https://github.com/10jqka-aicubes/project-demo)



## 反作弊声明

1）参与者不允许在比赛中抄袭他人作品、使用多个小号，经发现将取消成绩；

2）参与者禁止在指定考核技术能力的范围外利用规则漏洞或技术漏洞等途径提高成绩排名，经发现将取消成绩；

3）在A榜中，若主办方认为排行榜成绩异常，需要参赛队伍配合给出可复现的代码。



## 赛事交流

![同花顺比赛小助手](http://speech.10jqka.com.cn/arthmetic_operation/245984a4c8b34111a79a5151d5cd6024/客服微信.JPEG)