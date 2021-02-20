import tensorflow as tf

x = [[1, 2, 3], [4, 5, 6]]
with tf.Session() as sess:
    begin = [0, 1]  # 从x[0,1],即元素2开始抽取
    size = [2, 1]  # 从x[0,1]开始，对x的第一个维度（行）抽取2个元素，在对x的第二个维度（列）抽取1个元素
    print(sess.run(tf.slice(x, begin, size)))  # 输出[[2],[5]]
