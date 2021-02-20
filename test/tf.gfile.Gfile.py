import tensorflow as tf

#file = "test.txt"
file = "bert_config.json"
with tf.io.gfile.GFile(file, 'r') as reader:
    text = reader.read()
print(text)
