# command for create pretraining input-data (制作预训练所使用的tfrecord数据集)

python create_pretraining_data.py --input_file=./sample_text.txt --output_file=./tmp/tf_examples.tfrecord --vocab_file=vocab.txt --do_lower_case=True --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5

# command for pretraining (预训练)
python run_pretraining.py --input_file=./tmp/tf_examples.tfrecord --output_dir=./tmp/pretraining_output --do_train=true --do_eval=true --bert_config_file=bert_tiny/bert_config.json --init_checkpoint=bert_tiny/bert_model.ckpt --train_batch_size=32 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=5 --num_warmup_steps=2 --learning_rate=2e-5

# fine-tuning (微调)
## fine-tuning for classifier (针对分类任务的微调)
BERT_BASE_DIR=./bert_tiny
GLUE_DIR=./glue_data
TASK_NAME=CoLA
python run_classifier.py --task_name=$TASK_NAME --do_train=true --do_eval=true --data_dir=$GLUE_DIR/$TASK_NAME --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=/tmp/cola_output/