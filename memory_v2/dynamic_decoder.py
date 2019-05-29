# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Seq2seq layer operations for use in neural networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs

__all__ = ["dynamic_rnn_decoder"]

def dynamic_rnn_decoder(cell,  # 多层的 RNNCell
                        decoder_fn,  # 对输入输出进行处理的函数
                        inputs=None,  # 训练时传入该参数，为response的嵌入向量拼接使用的三元组 [batch_size, decoder_len, num_embed_units+3*num_trans_units]
                        sequence_length=None,  # 训练时传入该参数，为response的长度向量
                        parallel_iterations=None,  # 没用到这个参数
                        swap_memory=False,  # 没用到这个参数
                        time_major=False,  # 表示输入的数据集是否是time-major的
                        scope=None,  # "decoder_rnn"
                        name=None):  # 没用到这个参数
    """seq2seq模型的RNN动态解码器
    """
    with ops.name_scope(name, "dynamic_rnn_decoder",
                        [cell, decoder_fn, inputs, sequence_length,
                         parallel_iterations, swap_memory, time_major, scope]):
        # 训练时对输入进行处理
        if inputs is not None:
            inputs = ops.convert_to_tensor(inputs)
            if inputs.get_shape().ndims is not None and (
                    inputs.get_shape().ndims < 2):
                raise ValueError("Inputs must have at least two dimensions")

            # 如果不是time_major就做一个转置 [batch, seq, features] -> [seq, batch, features]
            if not time_major:
                inputs = array_ops.transpose(inputs, perm=[1, 0, 2])  # [decoder_len, batch_size, num_embed_units+3*num_trans_units]

            dtype = inputs.dtype
            input_depth = int(inputs.get_shape()[2])  # num_embed_units+3*num_trans_units
            batch_depth = inputs.get_shape()[1].value  # batch_size
            max_time = inputs.get_shape()[0].value  # decoder_len
            if max_time is None:
                max_time = array_ops.shape(inputs)[0]

            # 将解码器的输入设置成一个TensorArray，长度为decoder_len
            inputs_ta = tensor_array_ops.TensorArray(dtype, size=max_time)
            inputs_ta = inputs_ta.unstack(inputs)  # 数组的每个元素是个[batch_size, num_embed_units+3*num_trans_units]的张量

########解码器复写的循环函数                                                                                         ###
        def loop_fn(time, cell_output, cell_state, loop_state):
            """loop_fn 是一个函数，这个函数在 rnn 的相邻时间步之间被调用。
            函数的总体调用过程为：
            1. 初始时刻，先调用一次loop_fn，获取第一个时间步的cell的输入，loop_fn 中进行读取初始时刻的输入。
            2. 进行cell自环 (output, cell_state) = cell(next_input, state)
            3. 在t时刻RNN计算结束时，cell有一组输出cell_output和状态cell_state，都是tensor；
            4. 到t+1时刻开始进行计算之前，loop_fn被调用，调用的形式为
                loop_fn( t, cell_output, cell_state, loop_state)，而被期待的输出为：(finished, next_input, initial_state, emit_output, loop_state)；
            5. RNN采用loop_fn返回的next_input作为输入，initial_state作为状态，计算得到新的输出。
            在每次执行（output， cell_state） =  cell(next_input, state)后，执行loop_fn()进行数据的准备和处理。
            emit_structure即上文的emit_output将会按照时间存入emit_ta中。
            loop_state记录rnn loop的变量的状态。用作记录状态
            tf.where是用来实现dynamic的。
            time: 第time个时间步之前的处理，起始为0
            cell_output: 上一个时间步的输出
            cell_state: RNNCells 的长时记忆
            loop_state: 保存了上个时间步执行后是否已经结束，如果输出 alignments，还保存了存有alignments的TensorArray
            """

            # 取出循环状态
            if cell_state is None:  # time=0
                if cell_output is not None:
                    raise ValueError("Expected cell_output to be None when cell_state " 
                                     "is None, but saw: %s" % cell_output)
                if loop_state is not None:
                    raise ValueError("Expected loop_state to be None when cell_state " 
                                     "is None, but saw: %s" % loop_state)
                context_state = None
            else:  # time>=1
                if isinstance(loop_state, tuple):  # 如果记录了对齐
                    (done, context_state) = loop_state
                else:  # 如果没有记录对齐
                    done = loop_state  # done: [batch_size]为bool值标识了每个batch是否已经解码结束
                    context_state = None

            # 训练时
            if inputs is not None:
                if cell_state is None:  # time=0
                    next_cell_input = inputs_ta.read(0)
                else:  # time>=1
                    if batch_depth is not None:
                        batch_size = batch_depth
                    else:
                        batch_size = array_ops.shape(done)[0]
                    # 如果time == max_time解码结束, 则next_cell_input=[batch_size, input_depth]的全1矩阵
                    # 否则，next_cell_input读取这一时间步的输入
                    next_cell_input = control_flow_ops.cond(
                            math_ops.equal(time, max_time),
                            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtype),
                            lambda: inputs_ta.read(time))
                # next_done=None, emit_output=attention
                (next_done, next_cell_state, next_cell_input, emit_output, next_context_state) = \
                    decoder_fn(time, cell_state, next_cell_input, cell_output, context_state)
            # 推导时
            else:
                (next_done, next_cell_state, next_cell_input, emit_output, next_context_state) = \
                    decoder_fn(time, cell_state, None, cell_output, context_state)

            # 检查结束状态
            if next_done is None:  # 当训练时，next_done 返回的是 None
                next_done = time >= sequence_length  # 当 time >= sequence_length 时，next_done = True

            # 存储循环状态
            if next_context_state is None:  # 如果不输出alignments
                next_loop_state = next_done
            else:  # 如果输出alignments
                next_loop_state = (next_done, next_context_state)

            return (next_done, next_cell_input, next_cell_state,
                            emit_output, next_loop_state)
########                                                                                                             ###

        # Run raw_rnn function
        outputs_ta, final_state, final_loop_state = rnn.raw_rnn(cell,
                                                                loop_fn,
                                                                parallel_iterations=parallel_iterations,
                                                                swap_memory=swap_memory,
                                                                scope=scope)
        outputs = outputs_ta.stack()

        # 如果要输出alignments就获取final_context_state
        if isinstance(final_loop_state, tuple):
            final_context_state = final_loop_state[1]
        else:
            final_context_state = None

        # 转置回去
        if not time_major:
            # [seq, batch, features] -> [batch, seq, features]
            outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
        return outputs, final_state, final_context_state
