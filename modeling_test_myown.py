import tensorflow as tf
import modeling

input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
flat_token_type_ids = tf.reshape(token_type_ids, [-1])
one_hot_token_type_ids = tf.one_hot(flat_token_type_ids, depth=2)

config = modeling.BertConfig(vocab_size=32000, hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                             intermediate_size=1024, type_vocab_size=2)

model = modeling.BertModel(config=config, is_training=True, input_ids=input_ids, input_mask=input_mask,
                           token_type_ids=token_type_ids)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(one_hot_token_type_ids))
    print(sess.run(model.get_all_encoder_layers()))
