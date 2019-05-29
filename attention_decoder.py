from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


def attention_decoder_fn_train(encoder_state,
                               attention_keys,
                               attention_values,
                               attention_score_fn,
                               attention_construct_fn,
                               output_alignments=False,
                               max_length=None,  # tf.reduce_max(responses_length)
                               name=None):
    """构造一个训练时处理解码时每个时间步的输出和下一步的输入的函数
    """
    with ops.name_scope(name, "attention_decoder_fn_train", [
            encoder_state, attention_keys, attention_values, attention_score_fn,
            attention_construct_fn
    ]):
        pass

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        """处理每个时间步输出并准备下个时间步输入的函数
        Args:
            time: 记录时间步
            cell_state: RNNCell的长时记忆，在解码的第0时间步之前用编码器的输出状态初始化，之后都不用去管
            cell_input: 第time个时间步的输入
            cell_output: 第time个时间步上1个时间步的输出
            context_state: 用于存储一些自己想记录的数据，例如对齐
        """
        with ops.name_scope(name, "attention_decoder_fn_train", [time, cell_state, cell_input, cell_output, context_state]):
            # time=0的处理
            if cell_state is None:
                # 初始化解码器状态
                cell_state = encoder_state
                # 采用初始化的 attention
                attention = _init_attention(encoder_state)
                # 如果要输出alignments，则声明一个TensorArray用来记录
                if output_alignments:
                    context_state = tensor_array_ops.TensorArray(dtype=dtypes.float32,
                                                                 tensor_array_name="alignments_ta",
                                                                 size=max_length,
                                                                 dynamic_size=True,
                                                                 infer_shape=False)
            # time>=1的处理
            else:
                attention = attention_construct_fn(cell_output, attention_keys, attention_values)
                if output_alignments:
                    attention, alignments = attention
                    context_state = context_state.write(time-1, alignments)  # 记录一下 alignments
                cell_output = attention
            # 拼接cell_input和attention成为time时间步的输入
            next_input = array_ops.concat([cell_input, attention], 1)
            return (None, cell_state, next_input, cell_output, context_state)

    return decoder_fn


def attention_decoder_fn_inference(output_fn,
                                   encoder_state,
                                   attention_keys,
                                   attention_values,
                                   attention_score_fn,
                                   attention_construct_fn,
                                   embeddings,
                                   start_of_sequence_id,
                                   end_of_sequence_id,
                                   maximum_length,
                                   num_decoder_symbols,
                                   dtype=dtypes.int32,
                                   selector_fn=None,
                                   imem=None,
                                   name=None):
    """构造一个推导时处理解码时每个时间步的输出和下一步的输入的函数
    Args:
       imem:
            ([batch_size,triple_num*triple_len,num_embed_units], 实体嵌入
             [encoder_batch_size, triple_num*triple_len, 3*num_trans_units]) 三元组嵌入
    """
    with ops.name_scope(name, "attention_decoder_fn_inference", [
            output_fn, encoder_state, attention_keys, attention_values,
            attention_score_fn, attention_construct_fn, embeddings, imem,
            start_of_sequence_id, end_of_sequence_id, maximum_length,
            num_decoder_symbols, dtype
    ]):
        # 将一些数值转化成张量
        start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, dtype)
        end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
        maximum_length = ops.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)

        encoder_info = nest.flatten(encoder_state)[0]
        batch_size = encoder_info.get_shape()[0].value

        # 如果output_fn为None则做一个恒等变换
        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = array_ops.shape(encoder_info)[0]

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        """处理每个时间步输出并准备下个时间步输入的函数
        """
        with ops.name_scope(
                name, "attention_decoder_fn_inference",
                [time, cell_state, cell_input, cell_output, context_state]):
            # 推导时没有输入
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" % cell_input)
            # time=0
            if cell_output is None:
                # 下一步的输入
                next_input_id = array_ops.ones([batch_size,], dtype=dtype) * (start_of_sequence_id)  # [batch_size] start_of_sequence_id
                # 是否解码完成
                done = array_ops.zeros([batch_size,], dtype=dtypes.bool)  # [batch_size] False
                # 解码器状态初始化
                cell_state = encoder_state
                # 第0个时间步之前的解码器输出
                cell_output = array_ops.zeros([num_decoder_symbols], dtype=dtypes.float32)  # [num_decoder_symbols]
                # 下一步输入的id转化成嵌入
                word_input = array_ops.gather(embeddings, next_input_id)  # [batch_size, num_embed_units]

                # 解码器输入拼接了这一步使用的三元组
                # naf_triple_id = array_ops.zeros([batch_size, 2], dtype=dtype)  # [batch_size, 2] 0
                # imem[1]: [encoder_batch_size, triple_num*triple_len, 3*num_trans_units] 三元组嵌入
                # triple_input = array_ops.gather_nd(imem[1], naf_triple_id)  # [batch_size, 3*num_trans_units]
                # cell_input = array_ops.concat([word_input, triple_input], axis=1)  # [batch_size, num_embed_units+3*num_trans_units]
                cell_input = word_input

                # 初始化注意力
                attention = _init_attention(encoder_state)
                if imem is not None:  # 如果传入了实体嵌入和词嵌入
                    context_state = tensor_array_ops.TensorArray(dtype=dtypes.int32, tensor_array_name="output_ids_ta",
                                                                 size=maximum_length, dynamic_size=True,
                                                                 infer_shape=False)
            # time >= 1
            else:
                # 构建注意力
                attention = attention_construct_fn(cell_output, attention_keys, attention_values)
                if type(attention) is tuple:  # 输出了alignments
                    attention, alignment = attention
                    cell_output = attention
                    alignment = tf.reshape(alignment, [batch_size, -1])  # [batch_size, triple_num*triple_len]或者[batch_size, decoder_len]
                    selector = selector_fn(cell_output)  # 选择实体词的概率选择器
                    logit = output_fn(cell_output)  # [batch_size, num_decoder_symbols] 未softmax的预测
                    word_prob = nn_ops.softmax(logit) * (1 - selector)  # [batch_size, num_decoder_symbols] 选择生成词概率
                    entity_prob = alignment * selector  # 选择实体词的概率 [batch_size, triple_num*triple_len]或者[batch_size, decoder_len]

                    # [batch_size, 1] 该步是否选择生成词
                    # 1、tf.reduce_max(word_prob, 1): [batch_size] 生成词最大的概率
                    # 2、tf.reduce_max(entity_prob, 1): [batch_size] 实体词最大的概率
                    # 3、greater: [batch_size] 生成词的概率是否大于实体词概率
                    # 4、cast: [batch_size] 将bool值转化成浮点
                    # 5、reshape(cast): [batch_size, 1] 用生成词则为1，否则则为0
                    mask = array_ops.reshape(
                        math_ops.cast(math_ops.greater(tf.reduce_max(word_prob, 1), tf.reduce_max(entity_prob, 1)),
                                      dtype=dtypes.float32),
                        [-1, 1])

                    # [batch_size, num_embed_units] 当前时间步输入的嵌入
                    # 1、cast(math_ops.argmax(word_prob, 1): [batch_size] 生成词中最大概率的下标
                    # 2、gather: [batch_size， num_embed_units]: 采用的生成词
                    # 3、mask * gather: [batch_size, num_embed_units] 实际采用的生成词
                    # 4、reshape(range(batch_size)): [batch_size, 1]
                    # 5、reshape(cast(argmax(entity_prob, 1))): [batch_size, 1] 实体词中最大概率的下标
                    # 6、concat: [batch_size, 2] 4、5 两步的结果在第1维度上拼接
                    # 7、imem[0]:[batch_size, triple_num*triple_len, num_embed_units]
                    # 8、gather_nd: [batch_size, num_embed_units] 采用的实体词
                    # 9、(1-mask) * gather_nd: 实际采用的生成词
                    # 10、mask*gather+(1-mask)*gather_nd: [batch_size, num_embed_units] 当前时间步输入的嵌入
                    word_input = mask * array_ops.gather(embeddings, math_ops.cast(math_ops.argmax(word_prob, 1), dtype=dtype)) + \
                                 (1-mask)*array_ops.gather_nd(imem[0], array_ops.concat([array_ops.reshape(math_ops.range(batch_size, dtype=dtype), [-1, 1]),
                                                                                         array_ops.reshape(math_ops.cast(math_ops.argmax(entity_prob, 1), dtype=dtype), [-1, 1])],
                                                                                     axis=1))

                    # [batch_size, 2] 当前时间步选择实体词的索引
                    # 1、reshape(range(batch_size)): [batch_size, 1]
                    # 2、cast(1-mask): [batch_size, 1] 选择实体词的 mask
                    # 3、reshape(argmax(alignment, 1)): [batch_size, 1] 选择实体词的下标
                    # 4、cast(1-mask) * reshape(argmax(alignment, 1)): [batch_size, 1] 选择了实体词，则为实体词下标，否则则为0
                    # 5、concat: [batch_size, 2] 第二个维度的第一个元素为 batch，第二个元素为 indice
                    # indices = array_ops.concat([array_ops.reshape(math_ops.range(batch_size, dtype=dtype), [-1, 1]),
                    #                             math_ops.cast(1-mask, dtype=dtype) *
                    #                             tf.reshape(math_ops.cast(math_ops.argmax(alignment, 1), dtype=dtype), [-1, 1])],
                    #                            axis=1)
                    # imem[1]: [encoder_batch_size, triple_num*triple_len, 3*num_trans_units] 三元组嵌入
                    # 使用的三元组嵌入
                    # triple_input = array_ops.gather_nd(imem[1], indices)  # [batch_size, 3*num_trans_units]
                    # 当前时间步单词的嵌入拼上所用三元组的嵌入
                    # cell_input = array_ops.concat([word_input, triple_input], axis=1)  # [batch_size, num_embed_units+3*num_trans_units]
                    cell_input = word_input

                    mask = array_ops.reshape(math_ops.cast(mask, dtype=dtype), [-1])  # [batch_size] 选择生成词的 mask

                    # 当前时间步输入的单词id，如果为生成词则id为正，如果为实体词则id为负
                    # argmax(word_prob, 1): [batch_size] 生成词下标
                    # mask - 1: [batch_size] 如果取生成词则为 0，如果取实体词则为 -1
                    # argmax(entity_prob, 1): [batch_size] 实体词下标
                    # input_id: [batch_size] 如果为生成词则id为正，如果为实体词则id为负
                    input_id = mask * math_ops.cast(math_ops.argmax(word_prob, 1), dtype=dtype) + \
                               (mask - 1) * math_ops.cast(math_ops.argmax(entity_prob, 1), dtype=dtype)

                    # 把 input_id 写入 TensorArray
                    context_state = context_state.write(time-1, input_id)
                    # 判断句子是否已经结束
                    done = array_ops.reshape(math_ops.equal(input_id, end_of_sequence_id), [-1])
                    cell_output = logit  # [batch_size, num_decoder_symbols] 未softmax的预测
                else:  # 不输出 alignments 的情况
                    cell_output = attention

                    cell_output = output_fn(cell_output)  # [batch_size, num_decoder_symbols] 未softmax的预测
                    # [batch_size] 最大概率生成词的下标
                    next_input_id = math_ops.cast(
                            math_ops.argmax(cell_output, 1), dtype=dtype)
                    # 判断句子是否已经结束
                    done = math_ops.equal(next_input_id, end_of_sequence_id)
                    # 下个时间步细胞输入
                    cell_input = array_ops.gather(embeddings, next_input_id)  # [batch_size, num_embed_units]

            # 下个时间步输入，加上 attention
            next_input = array_ops.concat([cell_input, attention], 1)

            # 如果 time > maximum_length 则返回全为 True 的向量，否则返回 done
            done = control_flow_ops.cond(
                    math_ops.greater(time, maximum_length),
                    lambda: array_ops.ones([batch_size,], dtype=dtypes.bool),
                    lambda: done)
            return (done, cell_state, next_input, cell_output, context_state)

    return decoder_fn

def attention_decoder_fn_beam_inference(output_fn,  # 输出层给出的输出函数
                                       encoder_state,  # 编码器状态
                                       attention_keys,  # 注意力的key
                                       attention_values,  # 注意力的value
                                       attention_score_fn,  # 计算注意力分数的函数
                                       attention_construct_fn,  # 构造上下文的函数
                                       embeddings,  # 词嵌入
                                       start_of_sequence_id,  # 序列的起始id
                                       end_of_sequence_id,  # 序列的结束id
                                       maximum_length,  # 最大长度
                                       num_decoder_symbols,
                                       beam_size,
                                       remove_unk=False,
                                       d_rate=0.0,
                                       dtype=dtypes.int32,
                                       name=None):
    """推导时用于 dynamic_rnn_decoder 的注意力 decoder 函数

    attention_decoder_fn_inference 是一个用于 seq2seq 模型简单的推导函数。
    它能够在 dynamic_rnn_decoder 在推导模式下被使用。

    Returns:
        拥有 dynamic_rnn_decoder 所需接口的解码器函数
        用于推导。
    """
    with ops.name_scope(name, "attention_decoder_fn_inference", [
            output_fn, encoder_state, attention_keys, attention_values,
            attention_score_fn, attention_construct_fn, embeddings,
            start_of_sequence_id, end_of_sequence_id, maximum_length,
            num_decoder_symbols, dtype
    ]):
        # encoder：应该为[layer_num, batch_size, encoder_size]
        # with_rank()给定rank返回一个shape张量
        state_size = int(encoder_state[0].get_shape().with_rank(2)[1])  # encoder_size
        state = []
        for s in encoder_state:  # s:[batch_size, encoder_size]
            # reshape: s->[batch_size, 1, encoder_size]
            # concat: [batch_size, 5, encoder_size]
            # reshape: [batch_size*5, encoder_size]
            # state: [layer_num, batch_size, encoder_size]
            # 实现了将初始的状态复制5次的效果
            state.append(array_ops.reshape(array_ops.concat([array_ops.reshape(s, [-1, 1, state_size])]*beam_size, 1), [-1, state_size]))
        encoder_state = tuple(state)
        # attention_values：[batch_size, encoder_len, num_units]
        origin_batch = array_ops.shape(attention_values)[0]  # batch_size
        attn_length = array_ops.shape(attention_values)[1]  # encoder_len
        attention_values = array_ops.reshape(array_ops.concat([array_ops.reshape(attention_values, [-1, 1, attn_length, state_size])]*beam_size, 1), [-1, attn_length, state_size])
        attn_size = array_ops.shape(attention_keys)[2]
        attention_keys = array_ops.reshape(array_ops.concat([array_ops.reshape(attention_keys, [-1, 1, attn_length, attn_size])]*beam_size, 1), [-1, attn_length, attn_size])
        start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, dtype)
        end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
        maximum_length = ops.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)
        encoder_info = nest.flatten(encoder_state)[0]
        batch_size = encoder_info.get_shape()[0].value
        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = array_ops.shape(encoder_info)[0]
        #beam_size = ops.convert_to_tensor(beam_size, dtype)

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        """在 dynamic_rnn_decoder 中用于推导的解码器函数

        这个解码器函数和 attention_decoder_fn_train 中的 decoder_fn 最大的区别是，next_cell_input
        是如何计算的。在解码器函数中，我们通过在解码器输出的特征维度上使用一个 argmax 来计算下一个输入。
        这是一种 greedy-search 的方式。(Bahdanau et al., 2014) & (Sutskever et al., 2014) 使用 beam-search。

        Args:
            time: 反映当前时间步的正整型常量                     positive integer constant reflecting the current timestep.
            cell_state: RNNCell 的状态                          state of RNNCell.
            cell_input: dynamic_rnn_decoder 提供的输入          input provided by `dynamic_rnn_decoder`.
            cell_output: RNNCell的输出                          output of RNNCell.
            context_state: dynamic_rnn_decoder 提供的上下文状态  context state provided by `dynamic_rnn_decoder`.
        Returns:
            一个元组 (done, next state, next input, emit output, next context state)
            其中:
            done: 一个指示哪个句子已经达到 end_of_sequence_id 的布尔向量。
            被 dynamic_rnn_decoder 用来提早停止。当 time>maximum_length 时，
            一个所有元素都为 true 的布尔向量被返回。
            next state: `cell_state`, 这个解码器函数不修改给定的状态。
            next input: cell_output 的 argmax 的嵌入被用作 next_input
            emit output: 如果 output_fn is None，所提供的 cell_output 被返回。
                否则被用来在计算 next_input 和返回 cell_output 之前更新 cell_output。
            next context state: `context_state`, 这个解码器函数不修改给定的上下文状态。
                当使用，例如，beam search 时，上下文状态能够被修改。
        Raises:
            ValueError: if cell_input is not None.
        """
        with ops.name_scope(
                name, "attention_decoder_fn_inference",
                [time, cell_state, cell_input, cell_output, context_state]):
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" %
                                                 cell_input)
            if cell_output is None:
                # invariant that this is time == 0
                next_input_id = array_ops.ones(
                        [batch_size,], dtype=dtype) * (start_of_sequence_id)
                done = array_ops.zeros([batch_size,], dtype=dtypes.bool)
                cell_state = encoder_state
                cell_output = array_ops.zeros(
                        [num_decoder_symbols], dtype=dtypes.float32)
                cell_input = array_ops.gather(embeddings, next_input_id)

                # init attention
                attention = _init_attention(encoder_state)
                # init context state
                log_beam_probs = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name="log_beam_probs", size=maximum_length, dynamic_size=True, infer_shape=False)
                beam_parents = tensor_array_ops.TensorArray(dtype=dtypes.int32, tensor_array_name="beam_parents", size=maximum_length, dynamic_size=True, infer_shape=False)
                beam_symbols = tensor_array_ops.TensorArray(dtype=dtypes.int32, tensor_array_name="beam_symbols", size=maximum_length, dynamic_size=True, infer_shape=False)
                result_probs = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name="result_probs", size=maximum_length, dynamic_size=True, infer_shape=False)
                result_parents = tensor_array_ops.TensorArray(dtype=dtypes.int32, tensor_array_name="result_parents", size=maximum_length, dynamic_size=True, infer_shape=False)
                result_symbols = tensor_array_ops.TensorArray(dtype=dtypes.int32, tensor_array_name="result_symbols", size=maximum_length, dynamic_size=True, infer_shape=False)
                context_state = (log_beam_probs, beam_parents, beam_symbols, result_probs, result_parents, result_symbols)
            else:
                # construct attention
                attention = attention_construct_fn(cell_output, attention_keys,
                        attention_values)
                cell_output = attention

                # beam search decoder
                (log_beam_probs, beam_parents, beam_symbols, result_probs, result_parents, result_symbols) = context_state
                
                cell_output = output_fn(cell_output)    # logits
                cell_output = nn_ops.softmax(cell_output)
                

                cell_output = array_ops.split(cell_output, [2, num_decoder_symbols-2], 1)[1]

                tmp_output = array_ops.gather(cell_output, math_ops.range(origin_batch)*beam_size)

                probs = control_flow_ops.cond(
                        math_ops.equal(time, ops.convert_to_tensor(1, dtype)),
                        lambda: math_ops.log(tmp_output+ops.convert_to_tensor(1e-20, dtypes.float32)),
                        lambda: math_ops.log(cell_output+ops.convert_to_tensor(1e-20, dtypes.float32)) + array_ops.reshape(log_beam_probs.read(time-2), [-1, 1]))

                probs = array_ops.reshape(probs, [origin_batch, -1])
                best_probs, indices = nn_ops.top_k(probs, beam_size * 2)
                #indices = array_ops.reshape(indices, [-1])
                indices_flatten = array_ops.reshape(indices, [-1]) + array_ops.reshape(array_ops.concat([array_ops.reshape(math_ops.range(origin_batch)*((num_decoder_symbols-2)*beam_size), [-1, 1])]*(beam_size*2), 1), [origin_batch*beam_size*2])
                best_probs_flatten = array_ops.reshape(best_probs, [-1])

                symbols = indices_flatten % (num_decoder_symbols - 2)
                symbols = symbols + 2
                parents = indices_flatten // (num_decoder_symbols - 2)

                probs_wo_eos = best_probs + 1e5*math_ops.cast(math_ops.cast((indices%(num_decoder_symbols-2)+2)-end_of_sequence_id, dtypes.bool), dtypes.float32)
                
                best_probs_wo_eos, indices_wo_eos = nn_ops.top_k(probs_wo_eos, beam_size)

                indices_wo_eos = array_ops.reshape(indices_wo_eos, [-1]) + array_ops.reshape(array_ops.concat([array_ops.reshape(math_ops.range(origin_batch)*(beam_size*2), [-1, 1])]*beam_size, 1), [origin_batch*beam_size])

                _probs = array_ops.gather(best_probs_flatten, indices_wo_eos)
                _symbols = array_ops.gather(symbols, indices_wo_eos)
                _parents = array_ops.gather(parents, indices_wo_eos)


                log_beam_probs = log_beam_probs.write(time-1, _probs)
                beam_symbols = beam_symbols.write(time-1, _symbols)
                beam_parents = beam_parents.write(time-1, _parents)
                result_probs = result_probs.write(time-1, best_probs_flatten)
                result_symbols = result_symbols.write(time-1, symbols)
                result_parents = result_parents.write(time-1, parents)


                next_input_id = array_ops.reshape(_symbols, [batch_size])

                state_size = int(cell_state[0].get_shape().with_rank(2)[1])
                attn_size = int(attention.get_shape().with_rank(2)[1])
                state = []
                for j in cell_state:
                    state.append(array_ops.reshape(array_ops.gather(j, _parents), [-1, state_size]))
                cell_state = tuple(state)
                attention = array_ops.reshape(array_ops.gather(attention, _parents), [-1, attn_size])

                done = math_ops.equal(next_input_id, end_of_sequence_id)
                cell_input = array_ops.gather(embeddings, next_input_id)

            # combine cell_input and attention
            next_input = array_ops.concat([cell_input, attention], 1)

            # if time > maxlen, return all true vector
            done = control_flow_ops.cond(
                    math_ops.greater(time, maximum_length),
                    lambda: array_ops.ones([batch_size,], dtype=dtypes.bool),
                    lambda: array_ops.zeros([batch_size,], dtype=dtypes.bool))
            return (done, cell_state, next_input, cell_output, (log_beam_probs, beam_parents, beam_symbols, result_probs, result_parents, result_symbols))#context_state)

    return decoder_fn


def prepare_attention(attention_states,  # 编码器输出encoder_output: [batch_size, encoder_len, num_units]
                      attention_option,  # 'bahdanau'
                      num_units,
                      imem=None,
                      output_alignments=False,  # 训练时为 True
                      reuse=False):
    """为注意力准备 key/values/functions
    imem = (graph_embed, triples_embedding)
        graph_embed: [batch_size, triple_num, 2*num_trans_units] 静态图
        triples_embedding: [encoder_batch_size, triple_num, triple_len, 3*num_trans_units] 知识图三元组的嵌入
    返回:
        attention_keys:
        attention_values:
        attention_score_fn: 用来计算上下文的函数
        attention_construct_fn: 用来计算拼接后上下文的函数
    """
    # 初始化上下文的 attention_keys/attention_values: [batch_size, encoder_len, num_units]
    with variable_scope.variable_scope("attention_keys", reuse=reuse) as scope:
        attention_keys = layers.linear(
            attention_states, num_units, biases_initializer=None, scope=scope)
        attention_values = attention_states

    # graph_embed: [batch_size, triple_num, 2*num_trans_units] 静态图
    # triples_embedding: [encoder_batch_size, triple_num, triple_len, 3*num_trans_units] 知识图三元组的嵌入
    if imem is not None:  # 存在静态图或者动态图
        if type(imem) is tuple:  # 既存在静态图，又存在动态图
            # 初始化静态图的 key/value: [batch_size, triple_num, num_units]
            with variable_scope.variable_scope("imem_graph", reuse=reuse) as scope:
                attention_keys2, attention_states2 = array_ops.split(layers.linear(
                    imem[0], num_units*2, biases_initializer=None, scope=scope), [num_units, num_units], axis=2)
            # 初始化动态图的 key/value: [encoder_batch_size, triple_num, triple_len, num_units]
            with variable_scope.variable_scope("imem_triple", reuse=reuse) as scope:
                attention_keys3, attention_states3 = array_ops.split(layers.linear(
                    imem[1], num_units*2, biases_initializer=None, scope=scope), [num_units, num_units], axis=3)
            attention_keys = (attention_keys, attention_keys2, attention_keys3)
            attention_values = (attention_states, attention_states2, attention_states3)
        else:  # 只存在静态图
            # 初始化静态图的 key/value: [batch_size, triple_num, num_units]
            with variable_scope.variable_scope("imem", reuse=reuse) as scope:
                attention_keys2, attention_states2 = array_ops.split(layers.linear(
                    imem, num_units*2, biases_initializer=None, scope=scope), [num_units, num_units], axis=2)
            attention_keys = (attention_keys, attention_keys2)
            attention_values = (attention_states, attention_states2)
    if imem is None:  # 没有静态图或者三元组
        # 计算编码器每一步输出的注意力的函数
        attention_score_fn = _create_attention_score_fn("attention_score", num_units, attention_option, reuse)
    else:  # 有静态图和三元组
        # (计算编码器每一步输出的注意力的函数, 计算静态图或三元组注意力的函数)
        attention_score_fn = (_create_attention_score_fn("attention_score", num_units, attention_option, reuse),
                              _create_attention_score_fn("imem_score", num_units, "luong",
                                                         reuse, output_alignments=output_alignments))
    # 这个函数用来计算拼接完的上下文
    attention_construct_fn = _create_attention_construct_fn("attention_construct",
                                                            num_units,
                                                            attention_score_fn,
                                                            reuse)
    return (attention_keys, attention_values, attention_score_fn, attention_construct_fn)

#
def _init_attention(encoder_state):
    """
    返回:
        attn: 和编码器顶层最后输出相同维度的全零注意力向量
    """
    # encoder_state: [num_layers, batch_size, num_units] 是一个列表里面每层为一个张量: [batch_size, num_units]
    # 然后每一层的tensor放入一个列表 [tensor1, tensor2, tensor3,...]
    # 所有整个的维度看上去是[num_layers, batch_size, num_units]，但它不是这样一个维度的张量
    if isinstance(encoder_state, tuple):  # 其实不用判断，直接取最顶层就可以了
        top_state = encoder_state[-1]
    else:  # 单层的解码器，就不用选择了
        top_state = encoder_state

    # LSTM的编码器状态维度是[num_layers, 2, batch_size, num_units]
    # 我们取完最顶层的之后，第二维是元组(c, h)，c是rnn的隐藏状态，h是rnn的输出
    if isinstance(top_state, rnn_cell_impl.LSTMStateTuple):  # LSTM
        attn = array_ops.zeros_like(top_state.h)
    else:  # GRU的话则不存在大小为2的这个维度
        attn = array_ops.zeros_like(top_state)
    return attn


def _create_attention_construct_fn(name,
                                   num_units,
                                   attention_score_fn,  # 计算注意力的函数
                                   reuse):
    """
    返回:
        一个拼接计算完的上下文的函数
    """
    with variable_scope.variable_scope(name, reuse=reuse) as scope:
        def construct_fn(attention_query, attention_keys, attention_values):
            '''拼接几个注意力
            返回:
                attention: [batch_size, num_units] 拼接完几个注意力并做一次线性变化得到的输出
                alignments: 注意力系数
            '''
            alignments = None
            # 如果有静态图或者三元组
            if type(attention_score_fn) is tuple:
                # 计算编码器每一步输出的注意力
                context0 = attention_score_fn[0](attention_query, attention_keys[0], attention_values[0])
                # 如果只有静态图
                if len(attention_keys) == 2:
                    context1 = attention_score_fn[1](attention_query, attention_keys[1], attention_values[1])
                # 如果既有静态图还有三元组
                elif len(attention_keys) == 3:
                    context1 = attention_score_fn[1](attention_query, attention_keys[1:], attention_values[1:])

                if type(context1) is tuple:
                    # 只有静态图且要求输出对齐
                    if len(context1) == 2:
                        context1, alignments = context1
                        concat_input = array_ops.concat([attention_query, context0, context1], 1)
                    # 既有静态图还有三元组
                    elif len(context1) == 3:
                        context1, context2, alignments = context1
                        concat_input = array_ops.concat([attention_query, context0, context1, context2], 1)
                else:   # 存在静态图没有三元组且不要求输出静态图的对齐的情况
                    concat_input = array_ops.concat([attention_query, context0, context1], 1)
            # 如果没有静态图或者三元组
            else:
                context = attention_score_fn(attention_query, attention_keys, attention_values)
                concat_input = array_ops.concat([attention_query, context], 1)
            # 给拼接完的注意力做一个线性变化 [batch_size, num_units]
            attention = layers.linear(concat_input, num_units, biases_initializer=None, scope=scope)
            if alignments is None:
                return attention
            else:
                return attention, alignments
        return construct_fn


'''
    v: [num_units]
    keys: [batch_size, attention_length, attn_size]
    query: [batch_size, 1, attn_size]
    return weights [batch_size, attention_length]
'''
# 注意力计算方式1
# v内积tanh(keys+query)
@function.Defun(func_name="attn_add_fun", noinline=True)
def _attn_add_fun(v, keys, query):
    return math_ops.reduce_sum(v * math_ops.tanh(keys + query), [2])

# 注意力计算方式2
# keys内积query
@function.Defun(func_name="attn_mul_fun", noinline=True)
def _attn_mul_fun(keys, query):
    return math_ops.reduce_sum(keys * query, [2])

def _create_attention_score_fn(name,
                               num_units,
                               attention_option,
                               reuse,
                               output_alignments=False,  # 是否将 alignment 输出
                               dtype=dtypes.float32):
    """返回计算注意力的函数
    """
    with variable_scope.variable_scope(name, reuse=reuse):
        # 参数矩阵 query_w: [num_units, num_units]
        # 参数向量 score_v: [num_units]
        if attention_option == "bahdanau":
            query_w = variable_scope.get_variable(
                    "attnW", [num_units, num_units], dtype=dtype)
            score_v = variable_scope.get_variable("attnV", [num_units], dtype=dtype)

        def attention_score_fn(query, keys, values):
            """计算注意力分数和value的加权和
            Args:
                query: [batch_size, num_units] 上个时间步的输出
                keys: 不是元组时: [batch_size, encoder_len, num_unit]
                    是元组时: (graph_keys, triples_keys)
                               graph_keys: [batch_size, triple_num, num_unit] 静态图的key
                               triples_keys: [encoder_batch_size, triple_num, triple_len, num_unit] 三元组的key
                values: 不是元组时: [batch_size, encoder_len, num_units]
                    是元组时: (graph_values, triples_values)
                               graph_values: [batch_size, triple_num, num_unit] 静态图的value
                               triples_values: [encoder_batch_size, triple_num, triple_len, num_unit] 三元组的value
            """
            triple_keys, triple_values = None, None

            # 当 keys 为元组时(graph_keys, triples_keys)
            # keys 为静态图的key [batch_size, triple_num, num_units]
            # triple_keys 为三元组的key [batch_size, triple_num, triple_len, num_units]
            # values 为静态图的value [batch_size, triple_num, num_units]
            # triple_values 为三元组的value [batch_size, triple_num, triple_len, num_units]
            if type(keys) is tuple:
                keys, triple_keys = keys
                values, triple_values = values

            # 如果keys不为元组，则为解码器每一步输出的key [batch_size, encoder_len, num_unit]
            # 所以不管是解码器每一步输出还是静态图的key都可以统一成维度 [batch_size, attention_length, num_unit] 进行计算
            # 这两种方式可以用来计算对编码器每一步输出的注意力或静态图的注意力，但是不用于三元组的注意力计算
            if attention_option == "bahdanau":
                query = math_ops.matmul(query, query_w)  # 给query做一个线性变化 [batch_size, num_units]
                query = array_ops.reshape(query, [-1, 1, num_units])  # [batch_size, 1, num_units]
                # reduce_sum(score_v*tanh(keys+query), [2])
                scores = _attn_add_fun(score_v, keys, query)  # 注意力分数 [batch_size, attention_length]
            elif attention_option == "luong":  #
                query = array_ops.reshape(query, [-1, 1, num_units])  # [batch_size, 1, num_units]
                # reduce_sum(keys*query, [2])
                scores = _attn_mul_fun(keys, query)  # 注意力分数 [batch_size, attention_length]
            else:
                raise ValueError("Unknown attention option %s!" % attention_option)

            # alignments: softmax后的记忆力分数 [batch_size, attention_length]
            # TODO(thangluong): not normalize over padding positions.
            alignments = nn_ops.softmax(scores)

            # 计算通过注意力加权和的编码器输出或者静态图
            new_alignments = array_ops.expand_dims(alignments, 2)  # [batch_size, attention_length, 1]
            context_vector = math_ops.reduce_sum(new_alignments * values, [1])  # [batch_size, num_units]
            context_vector.set_shape([None, num_units])

            # 动态图的计算
            if triple_values is not None:
                # triple_keys: [batch_size, triple_num, triple_len, num_units]
                # luong方式计算对每个三元组的注意力分数 [batch_size, triple_num, triple_len]
                triple_scores = math_ops.reduce_sum(triple_keys*array_ops.reshape(query, [-1, 1, 1, num_units]), [3])
                triple_alignments = nn_ops.softmax(triple_scores)  # [batch_size, triple_num, triple_len]
                # 通过注意力对三元组的value求加权和 [batch_size, triple_num, num_units]
                context_triples = math_ops.reduce_sum(array_ops.expand_dims(triple_alignments, 3) * triple_values, [2])
                # 通过注意力对动态图求加权和 [batch_size, num_units]
                context_graph_triples = math_ops.reduce_sum(new_alignments * context_triples, [1])
                context_graph_triples.set_shape([None, num_units])

                # 对静态图的注意力*对三元组的注意力=实际对每个三元组的注意力
                final_alignments = new_alignments * triple_alignments  # [batch_size, triple_num, triple_len]
                return context_vector, context_graph_triples, final_alignments
            else:
                if output_alignments:
                    return context_vector, alignments  #
                else:
                    return context_vector  #
        return attention_score_fn
