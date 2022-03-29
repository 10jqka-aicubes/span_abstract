# !/bin/bash
# export TRAINED_CLASSIFIER=/home/yintaoye/nlp_research_3.0_20190909/data/answer_rank/ft_outputs/electra_qa_rank_1112_2
export TRAINED_CLASSIFIER=./result/ner/saved_model
# electra_qa_rank_pairwise_0925
#CUDA_VISIBLE_DEVICES=0 py2tf1.12python freeze_graph.py --input_saved_model_dir 1592291178/ --output_node_names Mean,loss/vjt50/dense_vjt50/Tanh,loss/vtiancheng/dense_vtiancheng/Tanh,loss/vjintou/dense_vjintou/Tanh,loss/vguojun/dense_vguojun/Tanh,loss/vfuguo/dense_vfuguo/Tanh --output_graph multitask_bert.pb
#CUDA_VISIBLE_DEVICES=1 py2tf1.12python freeze_graph.py --input_saved_model_dir 1597731983/ --output_node_names loss/Softmax --output_graph en_da_l4_h256.pb
#CUDA_VISIBLE_DEVICES=-1 py2tf1.12python freeze_graph.py --input_saved_model_dir 1598419248/ --output_node_names Mean --output_graph tc_std_ft_2l_40.pb
CUDA_VISIBLE_DEVICES=1 python freeze_graph.py --input_saved_model_dir=$TRAINED_CLASSIFIER/20220302/ --output_node_names loss_layer/outputs --output_graph $TRAINED_CLASSIFIER/pairwise_1115.pb

