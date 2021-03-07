export BERT_BASE_DIR='bert_tiny'
export GLUE_DIR='glue_data'
export TASK_NAME='CoLA'

python run_classifier.py \
	--task_name=$TASK_NAME \
	--do_train=false \
	--do_eval=false \
	--do_predict=true \
	--data_dir=$GLUE_DIR/$TASK_NAME \
	--vocab_file=$BERT_BASE_DIR/vocab.txt \
	--bert_config_file=$BERT_BASE_DIR/bert_config.json \
	--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
	--max_seq_length=128 \
	--train_batch_size=32 \
	--learning_rate=2e-5 \
	--num_train_epochs=3.0 \
	--output_dir=tmp/${TASK_NAME}_output/ \
	--do_export=true \
	--export_dir=exported/${TASK_NAME}