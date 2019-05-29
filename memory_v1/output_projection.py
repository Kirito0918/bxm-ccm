import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope

def output_projection_layer(num_units, num_symbols, num_samples=None, name="output_projection"):

    def output_fn(outputs):  # outputs: [batch_size, decoder_len, num_units]
        return layers.linear(outputs, num_symbols, scope=name)  # [batch_size, decoder_len, num_symbols]

    def selector_fn(outputs):  #
        selector = tf.sigmoid(layers.linear(outputs, 1, scope='selector'))
        return selector

    # 计算序列的交叉熵损失
    def sequence_loss(outputs, targets, masks):
        with variable_scope.variable_scope('decoder_rnn'):
            logits = layers.linear(outputs, num_symbols, scope=name)  # [batch_size, decoder_len, num_symbols]

            # 预测
            logits = tf.reshape(logits, [-1, num_symbols])  # [batch_size*decoder_len, num_symbols]
            # 标签
            local_labels = tf.reshape(targets, [-1])  # [batch_size*decoder_len]
            # 蒙版
            local_masks = tf.reshape(masks, [-1])  # [batch_size*decoder_len]
            
            local_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=local_labels, logits=logits)
            local_loss = local_loss * local_masks  # 序列长度外的部分不计算损失
            
            loss = tf.reduce_sum(local_loss)  # 序列的总损失
            total_size = tf.reduce_sum(local_masks)  # 序列的总长度
            total_size += 1e-12  # 避免总长度为0
            
            return loss / total_size  # 返回每个单词平均交叉熵损失

    # 采用了sampled softmax
    def sampled_sequence_loss(outputs, targets, masks):
        with variable_scope.variable_scope('decoder_rnn/%s' % name):
            weights = tf.transpose(tf.get_variable("weights", [num_units, num_symbols]))
            bias = tf.get_variable("biases", [num_symbols])
            local_labels = tf.reshape(targets, [-1, 1])
            local_outputs = tf.reshape(outputs, [-1, num_units])
            local_masks = tf.reshape(masks, [-1])
            local_loss = tf.nn.sampled_softmax_loss(weights, bias, local_labels,
                    local_outputs, num_samples, num_symbols)
            local_loss = local_loss * local_masks
            loss = tf.reduce_sum(local_loss)
            total_size = tf.reduce_sum(local_masks)
            total_size += 1e-12  # 避免总长度为0
            
            return loss / total_size
    
    def total_loss(outputs,  # [batch_size, decoder_len, num_units]
                   targets,  # [batch_size, decoder_len]
                   masks,  # [batch_size, decoder_len]
                   alignments,  # [batch_size, decoder_len, triple_num, triple_len]
                   triples_embedding,
                   use_entities,  # [batch_size, decoder_len] 用1标注了回复的哪个时间步用了三元组
                   entity_targets):  # [batch_size, decoder_len, triple_num, triple_len] 用1标注了每个batch每个时间步用了哪个图的哪个三元组

        batch_size = tf.shape(outputs)[0]
        local_masks = tf.reshape(masks, [-1])  # [batch_size*decoder_len]
        
        logits = layers.linear(outputs, num_symbols, scope='decoder_rnn/%s' % name)  # [batch_size, decoder_len, num_symbols]
        one_hot_targets = tf.one_hot(targets, num_symbols)  # [batch_size, decoder_len, num_symbols]

        # 每一步的单词预测正确的概率
        word_prob = tf.reduce_sum(tf.nn.softmax(logits) * one_hot_targets, axis=2)  # [batch_size, decoder_len]

        # 每一步使用实体词的概率预测
        selector = tf.squeeze(tf.sigmoid(layers.linear(outputs, 1, scope='decoder_rnn/selector')))  # [batch_size, decoder_len]

        # 每一步对使用的三元组的注意力
        triple_prob = tf.reduce_sum(alignments * entity_targets, axis=[2, 3])  # [batch_size, decoder_len]

        # 每一步正确的概率
        ppx_prob = word_prob * (1 - use_entities) + triple_prob * use_entities  # [batch_size, decoder_len]
        # 加上选择器选择正确的概率
        final_prob = word_prob * (1 - selector) * (1 - use_entities) + triple_prob * selector * use_entities  # [batch_size, decoder_len]

        # 加上选择器选择的损失
        final_loss = tf.reduce_sum(tf.reshape(- tf.log(1e-12 + final_prob), [-1]) * local_masks)
        # 不加选择器的ppx
        ppx_loss = tf.reduce_sum(tf.reshape( - tf.log(1e-12 + ppx_prob), [-1]) * local_masks)

        # 每个batch的ppx
        sentence_ppx = tf.reduce_sum(tf.reshape(tf.reshape(-tf.log(1e-12 + ppx_prob), [-1]) * local_masks, [batch_size, -1]), axis=1)  # [batch_size]
        # 选择器的损失
        selector_loss = tf.reduce_sum(tf.reshape(-tf.log(1e-12 + selector * use_entities + (1 - selector) * (1 - use_entities)), [-1]) * local_masks)  # [batch_size]
            
        loss = final_loss + selector_loss
        total_size = tf.reduce_sum(local_masks)
        total_size += 1e-12

        # 每个词的平均损失、每个词的平均ppx、每个样本的每个词的ppx[batch_size]
        return loss / total_size, ppx_loss / total_size, sentence_ppx / tf.reduce_sum(masks, axis=1)



    return output_fn, selector_fn, sequence_loss, sampled_sequence_loss, total_loss
    
