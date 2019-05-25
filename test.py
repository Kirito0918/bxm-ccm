import tensorflow as tf
import numpy as np

# with open('./data/resource.txt') as f:
#     d = json.loads(f.readline())
# raw_vocab = d['vocab_dict'] # 这是词汇的字典
# print(raw_vocab)
# vocab_list = sorted(raw_vocab, key=raw_vocab.get, reverse=True)
# print(vocab_list)

# dir.get应该是每个键的值
# dir = {"b": 200, "x": 300, "m": 301}
# print(dir.get)
# l = sorted(dir, key=dir.get, reverse=True)
# print(l)

# with open('./data/entity.txt') as f:
#     for i, line in enumerate(f):
#         e = line.strip()
#         print("%d:%s" % (i, line))

# NAF = ['_NAF_H', '_NAF_R', '_NAF_T']
# triple = [[NAF]]
# print(triple)

# str = 'i am bxm'
# print(str.split())

# print(list(range(5)))

# entity = [['_NONE']*12]
# print(entity)


# data = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# with tf.Session() as sess:
#     # print(type(sess.run(data)))
#     print(sess.run(data).tolist())
#     # print(list(sess.run(data)))
#     print(sess.run(tf.shape(data)))
#     print(sess.run(tf.unstack(tf.shape(data), axis=0)))

# match_triples = [
#     [[-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, 5, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1]],
#
#     [[-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, 4, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1]],
#
#     [[-1, -1, -1, -1, 3, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1]],
#
#     [[-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, 5, -1, -1, -1, -1]],
#
#     [[-1, -1, 8, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1]],
#
#     [[-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, 8, -1]],
#
#     [[-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1],
#      [-1, 1, -1, -1, -1, -1, -1]],
#
#     [[-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1, 8, -1]],
#     ]
# match_triples_arr = np.array(match_triples)
# with tf.Session() as sess:
#     one = tf.one_hot(match_triples_arr, 7)
#     # two = tf.reduce_sum(one, axis=(2, 3))
#     print(sess.run(one))
#     # print(sess.run(tf.shape(one)))

# a = tf.constant([[1], [1], [1]], dtype=tf.int32)
# b = tf.constant([[2, 3, 4], [2, 3, 4], [2, 3, 4]], dtype=tf.int32)
# c = tf.concat([a, tf.split(b, [2, 1], axis=1)[0]], 1)
# with tf.Session() as sess:
#     print(sess.run(a))
#     print(sess.run(b))
#     print(sess.run(c))

# a = np.array([1 , 1, 1, 1])
# print(a - 1)

# length = tf.constant([10, 18, 15, 9, 10, 13, 15], dtype=tf.int32)
# one = tf.one_hot(length - 1, 18)
# sum = tf.cumsum(one, reverse=True, axis=1)
# sh = tf.reshape(sum, [-1, 18])
# with tf.Session() as sess:
#     print(sess.run(length))
#     print('#############################')
#     print(sess.run(one))
#     print('#############################')
#     print(sess.run(sum))
#     print('#############################')
#     print(sess.run(sh))

# tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
# tf.app.flags.DEFINE_integer("symbols", 30000, "vocabulary size.")
# FLAGS = tf.app.flags.FLAGS
# # FLAGS._parse_flags()
# with tf.Session() as sess:
#     print(FLAGS.__flags)

# list = constant_op.constant(['a', 'b', 'c'])
# with tf.Session() as sess:
#     sess.run(list)
#     print(list)

# print(np.zeros((1, )))
# print(.0)
# previous_losses = [1e18]*3
# print(previous_losses)

# a =[1, 2, 3, 4, 5, 6]
# # random.shuffle(a)
# # print(a)
# # random.shuffle(a)
# # print(a)

# print(time.time())

# start = 0
# size = 32
# per = 10
# end = size * per
# print(list(range(start, end, size)))

# a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = tf.constant([2, 0, 1])
# with tf.Session() as sess:
#     print(sess.run(tf.nn.embedding_lookup(a, b)))

# a = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#                  [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
#                  [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
#                  ])
# b = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#                  [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
#                  [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
#                  ])
# with tf.Session() as sess:
#     c = tf.concat([a, b], axis=2)
#     print(sess.run(c))

# x = tf.constant([[1, 2, 3], [4, 5, 6]])
# with tf.Session() as sess:
#     print(sess.run(tf.transpose(x, perm=[1, 0])))


# a = tf.constant([[1, 2, 3],
#                  [4, 5, 6],
#                  [7, 8, 9]])
# x = tf.constant([list(range(5)), list(range(5))])
# op0 = tf.reduce_sum(x * x, axis=1)
# with tf.Session() as sess:
#     # print(sess.run(x * x))
#     # print(sess.run(a * a))
#     # print(sess.run(tf.multiply(a, a)))
#     # print(sess.run(tf.matmul(a, a, transpose_a=False, transpose_b=False)))
#     print(sess.run(tf.shape(x)))
#     print(sess.run(tf.shape(op0)))
#     print(sess.run(tf.shape(tf.expand_dims(op0, 1))))

# x = tf.constant(list(range(5)))
# y = tf.reshape(x, [-1, 1, 1])
# z = tf.tile(y, multiples=[1, 10, 1])
# with tf.Session() as sess:
#     print(sess.run(x))
#     print(sess.run(y))
#     print(sess.run(z))

# array = np.asarray([0, 1, 2], dtype=np.int32)
# a = tf.convert_to_tensor(array)
# print(a.get_shape().ndims)

# a = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
# ar = tf.TensorArray(dtype=tf.int32, dynamic_size=True, size=0)
# ar = ar.unstack(a)
# with tf.Session() as sess:
#     for time in range(len(a)):
#         print(sess.run(ar.read(time)))
#     ar = ar.stack()
#     print(sess.run(ar))


# tf.nn.rnn_cell_impl
# tf.zeros_like()

# list1 = [
#          [[1, 2, 3],
#           [4, 5, 6],
#           [7, 8, 9]],
#          [[11, 12, 13],
#           [14, 15, 16],
#           [17, 18, 19]],
#         ]
# list2 = [
#          [[1, 2, 3],
#           [4, 5, 6],
#           [7, 8, 9]],
#         ]
#
# a = tf.constant(list1)
# b = tf.constant(list2)
# print(list1 + list2)
# with tf.Session() as sess:
#     print(sess.run(a + b))
#     print(sess.run(a * b))


# list1 = [
#          [[1, 2, 3],
#           [4, 5, 6],
#           [7, 8, 9]],
#          [[11, 12, 13],
#           [14, 15, 16],
#           [17, 18, 19]],
#         ]
# list2 = [[1, 2, 3], [1, 2, 3]] # 2 * 3
# list3 = [[1], [2], [3]] # 3 * 1
#
# a = tf.constant(list1)
# b = tf.constant(list2)
# c = tf.constant(list3)
# with tf.Session() as sess:
#     print(sess.run(tf.matmul(c, b)))

# tf.expand_dims()

# a = tf.convert_to_tensor(np.reshape(np.array(range(27)), [3, 3, 3]))


# a = np.reshape(np.array(range(27)), [3, 3, 3])
# b = tf.get_variable(name='b', initializer=a)
# c = tf.reduce_sum(b, axis=[2])
# d = tf.placeholder(name='d', dtype=tf.int32)
# d.set_shape([None, 3, 3])
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(d, feed_dict={d: a}))


# a = tf.convert_to_tensor(np.reshape(np.array(range(9)), [3, 3]))
# b = tf.convert_to_tensor(np.reshape(np.array(range(27)), [3, 3, 3]))
# c = tf.expand_dims(a, axis=[2]) # 3 * 3 * 1
# with tf.Session() as sess:
#     print(sess.run(b))
#     print(sess.run(c))
#     print(sess.run(b * c))

# match_triples = [
#     [-1, -1, 3],
#     [-1, 2, -1],
#     [1, -1, -1]
# ]
# match_triples_arr = np.array(match_triples)
# with tf.Session() as sess:
#     one = tf.one_hot(match_triples_arr, 4)
#     print(sess.run(one))
#     print(sess.run(tf.shape(one)))

# list1 = list(range(10))
# print(list1[12])

# a = os.walk('./data')
# for root, dirs, files in a:
#     print(root)
#     print(dirs)
#     print(files)

# tf.nest.flatten()

# ones = tf.zeros([3,], dtype=tf.bool)
# two = tf.ones([3,], dtype=tf.bool)
# three = tf.cast(ones, dtype=tf.float32)
# four = tf.cast(two, dtype=tf.float32)
# with tf.Session() as sess:
#     print(sess.run(three))
#     print(sess.run(four))

# a = tf.constant(
#     [
#         [0.1, 0.3, 0.5],
#         [0.2, 0.4, 0.3],
#         [0.3, 0.6, 0.2],
#         [0.4, 0.1, 0.9],
#         [0.5, 0.3, 0.7]
#     ])
# b = tf.constant(
#     [
#         [0, 1],
#         [1, 2],
#         [2, 0]
#      ])
# # c = tf.gather(a, b)
# c = tf.gather_nd(a, b)
# with tf.Session() as sess:
#     print(sess.run(c))

# a = tf.range(10)
# b = tf.constant([0, 4, 8])
# with tf.Session() as sess:
#     print(sess.run(tf.gather(a, b)))

# a = tf.constant([[1, 2, 3], [4, 5, 6]])  # batch_size, memory_units
# b = tf.reshape(a, (2, 1, 3))
# c = tf.tile(b, multiples=(1, 3, 1))
# with tf.Session() as sess:
#     print(sess.run(c))

# a = tf.constant([[1, 2, 3], [4, 5, 6]])
# b = tf.constant([[3, 2, 1], [6, 5, 4]])
# c = []
# c.append(a)
# c.append(b)
# c = tf.reshape(tf.concat(c, axis=0), (2, 2, 3))
# with tf.Session() as sess:
#     print(sess.run(c))

print([1]*10)