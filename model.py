import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops.loss import sequence_loss
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.contrib.layers.python.layers import layers
from dynamic_decoder import dynamic_rnn_decoder
from output_projection import output_projection_layer
from attention_decoder import * 
from tensorflow.contrib.session_bundle import exporter

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
NONE_ID = 0
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class Model(object):
    def __init__(self,
            num_symbols,  # 词汇表size
            num_embed_units,  # 词嵌入size
            num_units,  # RNN 每层单元数
            num_layers,  # RNN 层数
            memory_units,  # 记忆网络向量的维度
            embed,  # 词嵌入
            entity_embed=None,  # 实体+关系的嵌入
            num_entities=0,  # 实体+关系的总个数
            num_trans_units=100,  # 实体嵌入的维度
            learning_rate=0.0001,  # 学习率
            learning_rate_decay_factor=0.95,  # 学习率衰退，并没有采用这种方式
            max_gradient_norm=5.0,  #
            num_samples=500,  # 样本个数，sampled softmax
            max_length=60,
            mem_use=True,
            output_alignments=True,
            use_lstm=False):
        
        self.posts = tf.placeholder(tf.string, (None, None), 'enc_inps')  # [batch_size, encoder_len]
        self.posts_length = tf.placeholder(tf.int32, (None), 'enc_lens')  # [batch_size]
        self.responses = tf.placeholder(tf.string, (None, None), 'dec_inps')  # [batch_size, decoder_len]
        self.responses_length = tf.placeholder(tf.int32, (None), 'dec_lens')  # [batch_size]
        self.entities = tf.placeholder(tf.string, (None, None, None), 'entities')  # [batch_size, triple_num, triple_len]
        self.entity_masks = tf.placeholder(tf.string, (None, None), 'entity_masks')  # 没用到
        self.triples = tf.placeholder(tf.string, (None, None, None, 3), 'triples')  # [batch_size, triple_num, triple_len, 3]
        self.posts_triple = tf.placeholder(tf.int32, (None, None, 1), 'enc_triples')  # [batch_size, encoder_len, 1]
        self.responses_triple = tf.placeholder(tf.string, (None, None, 3), 'dec_triples')  # [batch_size, decoder_len, 3]
        self.match_triples = tf.placeholder(tf.int32, (None, None, None), 'match_triples')  # [batch_size, decoder_len, triple_num]

        # 编码器batch_size，编码器encoder_len
        encoder_batch_size, encoder_len = tf.unstack(tf.shape(self.posts))
        triple_num = tf.shape(self.triples)[1]  # 知识图个数
        triple_len = tf.shape(self.triples)[2]  # 知识三元组个数

        # 使用的知识三元组
        one_hot_triples = tf.one_hot(self.match_triples, triple_len)  # [batch_size, decoder_len, triple_num, triple_len]
        # 用 1 标注了哪个时间步产生的回复用了知识三元组
        use_triples = tf.reduce_sum(one_hot_triples, axis=[2, 3])  # [batch_size, decoder_len]

        # 词汇映射到index的hash table
        self.symbol2index = MutableHashTable(
                key_dtype=tf.string,  # key张量的类型
                value_dtype=tf.int64,  # value张量的类型
                default_value=UNK_ID,  # 缺少key的默认值
                shared_name="in_table",  # If non-empty, this table will be shared under the given name across multiple sessions
                name="in_table",  # 操作名
                checkpoint=True)  # if True, the contents of the table are saved to and restored from checkpoints. If shared_name is empty for a checkpointed table, it is shared using the table node name.

        # index映射到词汇的hash table
        self.index2symbol = MutableHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.string,
                default_value='_UNK',
                shared_name="out_table",
                name="out_table",
                checkpoint=True)

        # 实体映射到index的hash table
        self.entity2index = MutableHashTable(
                key_dtype=tf.string,
                value_dtype=tf.int64,
                default_value=NONE_ID,
                shared_name="entity_in_table",
                name="entity_in_table",
                checkpoint=True)

        # index映射到实体的hash table
        self.index2entity = MutableHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.string,
                default_value='_NONE',
                shared_name="entity_out_table",
                name="entity_out_table",
                checkpoint=True)

        self.posts_word_id = self.symbol2index.lookup(self.posts)  # [batch_size, encoder_len]
        self.posts_entity_id = self.entity2index.lookup(self.posts)  # [batch_size, encoder_len]

        self.responses_target = self.symbol2index.lookup(self.responses)  # [batch_size, decoder_len]
        # 获得解码器的batch_size，decoder_len
        batch_size, decoder_len = tf.shape(self.responses)[0], tf.shape(self.responses)[1]
        # 去掉responses_target的最后一列，给第一列加上GO_ID
        self.responses_word_id = tf.concat([tf.ones([batch_size, 1], dtype=tf.int64)*GO_ID,
            tf.split(self.responses_target, [decoder_len-1, 1], 1)[0]], 1)  # [batch_size, decoder_len]

        # 得到response的mask
        self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1, 
            decoder_len), reverse=True, axis=1), [-1, decoder_len])  # [batch_size, decoder_len]

        # 初始化词嵌入和实体嵌入，传入了参数就直接赋值，没有的话就随机初始化
        if embed is None:
            self.embed = tf.get_variable('word_embed', [num_symbols, num_embed_units], tf.float32)
        else:
            self.embed = tf.get_variable('word_embed', dtype=tf.float32, initializer=embed)
        if entity_embed is None:  # 实体嵌入不随着模型的训练而更新
            self.entity_trans = tf.get_variable('entity_embed', [num_entities, num_trans_units], tf.float32, trainable=False)
        else:
            self.entity_trans = tf.get_variable('entity_embed', dtype=tf.float32, initializer=entity_embed, trainable=False)

        # 将实体嵌入传入一个全连接层
        self.entity_trans_transformed = tf.layers.dense(self.entity_trans, num_trans_units, activation=tf.tanh, name='trans_transformation')
        # 添加['_NONE', '_PAD_H', '_PAD_R', '_PAD_T', '_NAF_H', '_NAF_R', '_NAF_T']这7个的嵌入
        padding_entity = tf.get_variable('entity_padding_embed', [7, num_trans_units], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.entity_embed = tf.concat([padding_entity, self.entity_trans_transformed], axis=0)

        # triples_embedding: [batch_size, triple_num, triple_len, 3*num_trans_units] 知识图三元组的嵌入
        triples_embedding = tf.reshape(tf.nn.embedding_lookup(self.entity_embed, self.entity2index.lookup(self.triples)),
                                       [encoder_batch_size, triple_num, -1, 3 * num_trans_units])
        # entities_word_embedding: [batch_size, triple_num*triple_len, num_embed_units] 知识图中用到的所有实体的嵌入
        entities_word_embedding = tf.reshape(tf.nn.embedding_lookup(self.embed, self.symbol2index.lookup(self.entities)),
                                             [encoder_batch_size, -1, num_embed_units])
        # 分离知识图三元组的头、关系和尾 [batch_size, triple_num, triple_len, num_trans_units]
        head, relation, tail = tf.split(triples_embedding, [num_trans_units] * 3, axis=3)

        # 静态图注意力机制
        with tf.variable_scope('graph_attention'):
            # 将头尾连接起来 [batch_size, triple_num, triple_len, 2*num_trans_units]
            head_tail = tf.concat([head, tail], axis=3)
            # 将头尾送入全连接层 [batch_size, triple_num, triple_len, num_trans_units]
            head_tail_transformed = tf.layers.dense(head_tail, num_trans_units, activation=tf.tanh, name='head_tail_transform')
            # 将关系送入全连接层 [batch_size, triple_num, triple_len, num_trans_units]
            relation_transformed = tf.layers.dense(relation, num_trans_units, name='relation_transform')
            # 求头尾和关系两个向量的内积，获得对三元组的注意力系数
            e_weight = tf.reduce_sum(relation_transformed * head_tail_transformed, axis=3)  # [batch_size, triple_num, triple_len]
            alpha_weight = tf.nn.softmax(e_weight)  # [batch_size, triple_num, triple_len]
            # tf.expand_dims 使 alpha_weight 维度+1 [batch_size, triple_num, triple_len, 1]
            # 对第2个维度求和,由此产生静态图的向量表示
            graph_embed = tf.reduce_sum(
                tf.expand_dims(alpha_weight, 3) * head_tail, axis=2)  # [batch_size, triple_num, 2*num_trans_units]

        """graph_embed_input
        1、首先一维的range列表[0, 1, 2... encoder_batch_size个]转化成三维的[encoder_batch_size, 1, 1]的矩阵
        [[[0]], [[1]], [[2]],...]
        2、然后tf.tile将矩阵的第1维复制encoder_len遍，变成[encoder_batch_size， encoder_len， 1]
        [[[0],[0]...]],...]
        3、与posts_triple: [batch_size, encoder_len, 1]在第2维上进行拼接，形成一个indices: [batch_size, encoder_len, 2]矩阵，
        indices矩阵：
        [
         [[0 0], [0 0], [0 0], [0 0], [0 1], [0 0], [0 2], [0 0],...encoder_len],
         [[1 0], [1 0], [1 0], [1 0], [1 1], [1 0], [1 2], [1 0],...encoder_len],
         [[2 0], [2 0], [2 0], [2 0], [2 1], [2 0], [2 2], [2 0],...encoder_len]
         ,...batch_size
        ]
        4、tf.gather_nd根据索引检索graph_embed: [batch_size, triple_num, 2*num_trans_units]再回填至indices矩阵
        indices矩阵最后一个维度是2，例如有[0, 2]，表示这个时间步第1个batch用了第2个图，
        则找到这个知识图的静态图向量填入到indices矩阵的[0, 2]位置最后得到结果维度
        [encoder_batch_size, encoder_len, 2*num_trans_units]表示每个时间步用的静态图向量
        """
        # graph_embed_input = tf.gather_nd(graph_embed, tf.concat(
        #     [tf.tile(tf.reshape(tf.range(encoder_batch_size, dtype=tf.int32), [-1, 1, 1]), [1, encoder_len, 1]),
        #      self.posts_triple],
        #     axis=2))

        # 将responses_triple转化成实体嵌入 [batch_size, decoder_len, 300]，标识了response每个时间步用了哪个三元组的嵌入
        # triple_embed_input = tf.reshape(
        #     tf.nn.embedding_lookup(self.entity_embed, self.entity2index.lookup(self.responses_triple)),
        #     [batch_size, decoder_len, 3 * num_trans_units])

        post_word_input = tf.nn.embedding_lookup(self.embed, self.posts_word_id)  # [batch_size, encoder_len, num_embed_units]
        response_word_input = tf.nn.embedding_lookup(self.embed, self.responses_word_id)  # [batch_size, decoder_len, num_embed_units]

        # post_word_input和graph_embed_input拼接构成编码器输入 [batch_size, encoder_len, num_embed_units+2*num_trans_units]
        # self.encoder_input = tf.concat([post_word_input, graph_embed_input], axis=2)
        # response_word_input和triple_embed_input拼接构成解码器输入 [batch_size, decoder_len, num_embed_units+3*num_trans_units]
        # self.decoder_input = tf.concat([response_word_input, triple_embed_input], axis=2)

        encoder_cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])
        decoder_cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])
        
        # rnn encoder
        # encoder_state: [num_layers, 2, batch_size, num_units] 编码器输出状态 LSTM GRU:[num_layers, batch_size, num_units]
        encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, post_word_input,
                self.posts_length, dtype=tf.float32, scope="encoder")

########记忆网络                                                                                                     ###
        response_encoder_cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])
        response_encoder_output, response_encoder_state = tf.nn.dynamic_rnn(response_encoder_cell,
                                                                            response_word_input,
                                                                            self.responses_length,
                                                                            dtype=tf.float32,
                                                                            scope="response_encoder")

        # graph_embed: [batch_size, triple_num, 200] 静态图向量
        # encoder_state: [num_layers, batch_size, num_units]
        with tf.variable_scope("post_memory_network"):
            # 将静态知识图转化成输入向量m
            post_input = tf.layers.dense(graph_embed, memory_units, use_bias=False, name="post_weight_a")
            post_input = tf.tile(tf.reshape(post_input, (1, encoder_batch_size, triple_num, memory_units)),
                                 multiples=(num_layers, 1, 1, 1))  # [num_layers, batch_size, triple_num, memory_units]
            # 将静态知识库转化成输出向量c
            post_output = tf.layers.dense(graph_embed, memory_units, use_bias=False, name="post_weight_c")
            post_output = tf.tile(tf.reshape(post_output, (1, encoder_batch_size, triple_num, memory_units)),
                                 multiples=(num_layers, 1, 1, 1))  # [num_layers, batch_size, triple_num, memory_units]
            # 将question转化成状态向量u
            encoder_hidden_state = tf.reshape(tf.concat(encoder_state, axis=0), (num_layers, encoder_batch_size, num_units))
            post_state = tf.layers.dense(encoder_hidden_state, memory_units, use_bias=False, name="post_weight_b")
            post_state = tf.tile(tf.reshape(post_state, (num_layers, encoder_batch_size, 1, memory_units)),
                                 multiples=(1, 1, triple_num, 1))  # [num_layers, batch_size, triple_num, memory_units]
            # 概率p
            post_p = tf.reshape(tf.nn.softmax(tf.reduce_sum(post_state * post_input, axis=3)),
                                (num_layers, encoder_batch_size, triple_num, 1))  # [num_layers, batch_size, triple_num, 1]
            # 输出o
            post_o = tf.reduce_sum(post_output*post_p, axis=2)  # [num_layers, batch_size, memory_units]
            post_xstar = tf.concat([tf.layers.dense(post_o, memory_units, use_bias=False, name="post_weight_r"),
                                    encoder_state], axis=2)  # [num_layers, batch_size, num_units+memory_units]

        with tf.variable_scope("response_memory_network"):
            # 将静态知识图转化成输入向量m
            response_input = tf.layers.dense(graph_embed, memory_units, use_bias=False, name="response_weight_a")
            response_input = tf.tile(tf.reshape(response_input, (1, batch_size, triple_num, memory_units)),
                                     multiples=(num_layers, 1, 1, 1))  # [num_layers, batch_size, triple_num, memory_units]
            # 将静态知识库转化成输出向量c
            response_output = tf.layers.dense(graph_embed, memory_units, use_bias=False, name="response_weight_c")
            response_output = tf.tile(tf.reshape(response_output, (1, batch_size, triple_num, memory_units)),
                                      multiples=(num_layers, 1, 1, 1))  # [batch_size, triple_num, memory_units]
            # 将question转化成状态向量u
            response_hidden_state = tf.reshape(tf.concat(response_encoder_state, axis=0), (num_layers, batch_size, num_units))
            response_state = tf.layers.dense(response_hidden_state, memory_units, use_bias=False, name="response_weight_b")
            response_state = tf.tile(tf.reshape(response_state, (num_layers, batch_size, 1, memory_units)),
                                     multiples=(1, 1, triple_num, 1))  # [num_layers, batch_size, triple_num, memory_units]
            # 概率p
            response_p = tf.reshape(tf.nn.softmax(tf.reduce_sum(response_state * response_input, axis=3)),
                                (num_layers, batch_size, triple_num, 1))  # [num_layers, batch_size, triple_num, 1]
            # 输出o
            response_o = tf.reduce_sum(response_output*response_p, axis=2)  # [num_layers, batch_size, memory_units]
            response_ystar = tf.concat([tf.layers.dense(response_o, memory_units, use_bias=False, name="response_weight_r"),
                                        response_encoder_state], axis=2)  # [num_layers, batch_size, num_units+memory_units]

        with tf.variable_scope("memory_network"):
            memory_hidden_state = tf.layers.dense(tf.concat([post_xstar, response_ystar], axis=2),
                                                  num_units, use_bias=False, activation=tf.tanh, name="output_weight")
            # [num_layers, batch_size, num_units]
            memory_hidden_state = tf.split(memory_hidden_state, [1]*num_layers, axis=0)
########                                                                                                             ###

        output_fn, selector_fn, sequence_loss, sampled_sequence_loss, total_loss =\
            output_projection_layer(num_units, num_symbols, num_samples)
        
########用于训练的decoder                                                                                            ###
        with tf.variable_scope('decoder'):
            attention_keys_init, attention_values_init, attention_score_fn_init, attention_construct_fn_init \
                    = prepare_attention(encoder_output,
                                        'bahdanau',
                                        num_units,
                                        imem=(graph_embed, triples_embedding),
                                        output_alignments=output_alignments and mem_use)

            # 训练时处理每个时间步输出和下个时间步输入的函数
            decoder_fn_train = attention_decoder_fn_train(memory_hidden_state,
                                                          attention_keys_init,
                                                          attention_values_init,
                                                          attention_score_fn_init,
                                                          attention_construct_fn_init,
                                                          output_alignments=output_alignments and mem_use,
                                                          max_length=tf.reduce_max(self.responses_length))

            self.decoder_output, _, alignments_ta = dynamic_rnn_decoder(decoder_cell,
                                                                        decoder_fn_train,
                                                                        response_word_input,
                                                                        self.responses_length,
                                                                        scope="decoder_rnn")

            if output_alignments:
                self.decoder_loss, self.ppx_loss, self.sentence_ppx = total_loss(self.decoder_output,
                                                                                 self.responses_target,
                                                                                 self.decoder_mask,
                                                                                 self.alignments,
                                                                                 triples_embedding,
                                                                                 use_triples,
                                                                                 one_hot_triples)
                self.sentence_ppx = tf.identity(self.sentence_ppx, name='ppx_loss')
            else:
                self.decoder_loss = sequence_loss(self.decoder_output, self.responses_target, self.decoder_mask)
########                                                                                                             ###
########用于推导的decoder                                                                                            ###
        with tf.variable_scope('decoder', reuse=True):
            attention_keys, attention_values, attention_score_fn, attention_construct_fn \
                    = prepare_attention(encoder_output,
                                        'bahdanau',
                                        num_units,
                                        reuse=True,
                                        imem=(graph_embed, triples_embedding),
                                        output_alignments=output_alignments and mem_use)

            decoder_fn_inference = \
                attention_decoder_fn_inference(output_fn,
                                               encoder_state,
                                               attention_keys,
                                               attention_values,
                                               attention_score_fn,
                                               attention_construct_fn,
                                               self.embed,
                                               GO_ID,
                                               EOS_ID,
                                               max_length,
                                               num_symbols,
                                               imem=(entities_word_embedding,
                                                     tf.reshape(triples_embedding, [encoder_batch_size, -1, 3*num_trans_units])),
                                               selector_fn=selector_fn)
            # imem: ([batch_size,triple_num*triple_len,num_embed_units],
            # [encoder_batch_size, triple_num*triple_len, 3*num_trans_units]) 实体词嵌入和三元组嵌入的元组
                
            self.decoder_distribution, _, output_ids_ta = dynamic_rnn_decoder(decoder_cell,
                    decoder_fn_inference, scope="decoder_rnn")

            output_len = tf.shape(self.decoder_distribution)[1]  # decoder_len
            output_ids = tf.transpose(output_ids_ta.gather(tf.range(output_len)))  # [batch_size, decoder_len]

            # 对 output 的值域行裁剪
            word_ids = tf.cast(tf.clip_by_value(output_ids, 0, num_symbols), tf.int64)  # [batch_size, decoder_len]

            # 计算的是采用的实体词在 entities 的位置
            # 1、tf.shape(entities_word_embedding)[1] = triple_num*triple_len
            # 2、tf.range(encoder_batch_size): [batch_size]
            # 3、tf.reshape(tf.range(encoder_batch_size) * tf.shape(entities_word_embedding)[1], [-1, 1]): [batch_size, 1] 实体词在 entities 中的偏移量
            # 4、tf.clip_by_value(-output_ids, 0, num_symbols): [batch_size, decoder_len] 实体词的相对位置
            # 5、entity_ids: [batch_size * decoder_len] 加上偏移量之后在 entities 中的实际位置
            entity_ids = tf.reshape(tf.clip_by_value(-output_ids, 0, num_symbols) + tf.reshape(tf.range(encoder_batch_size) * tf.shape(entities_word_embedding)[1], [-1, 1]), [-1])

            # 计算的是所用的实体词
            # 1、entities: [batch_size, triple_num, triple_len]
            # 2、tf.reshape(self.entities, [-1]): [batch_size * triple_num * triple_len]
            # 3、tf.gather: [batch_size*decoder_len]
            # 4、entities: [batch_size, output_len]
            entities = tf.reshape(tf.gather(tf.reshape(self.entities, [-1]), entity_ids), [-1, output_len])

            words = self.index2symbol.lookup(word_ids)  # 将 id 转化为实际的词
            # output_ids > 0 为 bool 张量，True 的位置用 words 中该位置的词替换
            self.generation = tf.where(output_ids > 0, words, entities)
            self.generation = tf.identity(self.generation, name='generation')
########                                                                                                             ###

        # 初始化训练过程
        self.learning_rate = tf.Variable(float(learning_rate), 
                trainable=False, dtype=tf.float32)

        # 并没有使用衰退的学习率
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)

        # 更新参数的次数
        self.global_step = tf.Variable(0, trainable=False)

        # 要训练的参数
        self.params = tf.global_variables()

        # 选择优化算法
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.lr = opt._lr

        # 根据 decoder_loss 计算 params 梯度
        gradients = tf.gradients(self.decoder_loss, self.params)
        # 梯度裁剪
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
                global_step=self.global_step)

        tf.summary.scalar('decoder_loss', self.decoder_loss)
        for each in tf.trainable_variables():
            tf.summary.histogram(each.name, each)

        self.merged_summary_op = tf.summary.merge_all()
        
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, 
                max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        self.saver_epoch = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000, pad_step_number=True)

    # 打印参数
    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    #
    def step_decoder(self, session, data, forward_only=False, summary=False):
        input_feed = {self.posts: data['posts'],
                self.posts_length: data['posts_length'],
                self.responses: data['responses'],
                self.responses_length: data['responses_length'],
                self.triples: data['triples'],
                self.posts_triple: data['posts_triple'],
                self.responses_triple: data['responses_triple'],
                self.match_triples: data['match_triples']}

        if forward_only:
            output_feed = [self.sentence_ppx]
        else:
            output_feed = [self.sentence_ppx, self.gradient_norm, self.update]

        if summary:
            output_feed.append(self.merged_summary_op)

        return session.run(output_feed, input_feed)
