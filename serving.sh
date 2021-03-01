export BERT_BASE_DIR=bert_tiny
export TASK_NAME=CoLA

bert-base-serving-start \
	-bert_model_dir $BERT_BASE_DIR \
	-model_dir tmp/${TASK_NAME}_output/ \
	-model_pb_dir exported/${TASK_NAME}/1614598014/ \
	-mode CLASS \
	-max_seq_len 128 \
	-http_port 8091 \
	-port 5575 \
	-port_out 5576 \
	-device_map 1
