# use bert-base-serving to deploy bert
export BERT_BASE_DIR=bert_tiny
export TASK_NAME=CoLA
TASK_NAME=$(echo $TASK_NAME | tr '[A-Z]' '[a-z]')
export model_version=1614855916

bert-base-serving-start \
	-bert_model_dir $BERT_BASE_DIR \
	-model_dir models/${TASK_NAME}/ \
	-model_pb_dir models/${TASK_NAME}/${model_version}/ \
	-mode CLASS \
	-max_seq_len 128 \
	-http_port 8091 \
	-port 5575 \
	-port_out 5576
