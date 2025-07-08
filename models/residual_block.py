import tensorflow as tf
from typing import Optional
from models.inject_utils import is_input_target, is_weight_target, is_output_target
from models.inject_layers import InjectConv2D, InjectDense, BackwardInjectConv2D

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 strides=1,
                 is_first_block_of_first_layer=False,
                 l_name: Optional[str] = None,
                 seed=123,
                 inj_args=None):
        super(BasicBlock, self).__init__()
        self.l_name = l_name if l_name is not None else ''
        self.is_first_block_of_first_layer = is_first_block_of_first_layer
        if is_first_block_of_first_layer :
            self.conv1 = InjectConv2D(filters=filters,
                                      kernel_size=(3, 3),
                                      strides=strides,
                                      padding='same',
                                      l_name=self.l_name + '_conv1',
                                      seed=seed)
        else:
            self.conv1 = InjectConv2D(filters=filters,
                                      kernel_size=(3, 3),
                                      strides=strides,
                                      padding='same',
                                      l_name=self.l_name + '_conv1',
                                      seed=seed)

        self.bn1 = tf.keras.layers.BatchNormalization(name=self.l_name + '_bn1')
        self.relu1 = tf.keras.layers.ReLU(name=self.l_name + '_relu1')
        self.conv2 = InjectConv2D(filters=filters,
                                  kernel_size=(3, 3),
                                  strides=1,
                                  padding='same',
                                  l_name=self.l_name + '_conv2',
                                  seed=seed)
        self.bn2 = tf.keras.layers.BatchNormalization(name=self.l_name + '_bn2')
        if not is_first_block_of_first_layer:
            self.shortcut = InjectConv2D(filters=filters,
                                           kernel_size=(1, 1),
                                           strides=strides,
                                           padding='same',
                                           l_name=self.l_name + '_shortcut',
                                           seed=seed)
        self.add = tf.keras.layers.Add(name=self.l_name + '_add')
        self.relu2 = tf.keras.layers.ReLU(name=self.l_name + '_relu2')
        self.seed = seed

    def call(self, inputs, training=None, inject=None, inj_args=None):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}

        layer_inputs[self.l_name + '_conv1'] = inputs
        conv1_out, raw_conv1_out = self.conv1(inputs, inject=inject, inj_args=inj_args)
        layer_kernels[self.l_name + '_conv1'] = self.conv1.weights
        layer_outputs[self.l_name + '_conv1'] = raw_conv1_out

        bn1_out = self.bn1(conv1_out, training=training)
        relu1_out = self.relu1(bn1_out)

        layer_inputs[self.l_name + '_conv2'] = relu1_out
        conv2_out, raw_conv2_out = self.conv2(relu1_out, inject=inject, inj_args=inj_args)
        layer_kernels[self.l_name + '_conv2'] = self.conv2.weights
        layer_outputs[self.l_name + '_conv2'] = raw_conv2_out

        bn2_out = self.bn2(conv2_out, training=training)

        if not self.is_first_block_of_first_layer:
            layer_inputs[self.l_name + '_shortcut'] = inputs
            shortcut_out, raw_shortcut_out = self.shortcut(inputs, inject=inject, inj_args=inj_args)
            layer_kernels[self.l_name + '_shortcut'] = self.shortcut.weights
            layer_outputs[self.l_name + '_shortcut'] = raw_shortcut_out

            output = self.add([bn2_out, shortcut_out])
        else:
            output = self.add([bn2_out, inputs])

        output = self.relu2(output)
        outputs = {'add': output}

        return outputs, layer_inputs, layer_kernels, layer_outputs


class BasicBlocks(tf.keras.layers.Layer):
    def __init__(self, filters, strides, num_blocks, l_name: Optional[str] = None, seed=123):
        super(BasicBlocks, self).__init__()
        self.l_name = l_name if l_name is not None else ''
        self.blocks = tf.keras.Sequential()
        self.blocks.add(
            BasicBlock(filters=filters,
                       strides=strides,
                       is_first_block_of_first_layer=True,
                       l_name=self.l_name + '_block0'))

        for i in range(1, num_blocks):
            self.blocks.add(
                BasicBlock(filters=filters,
                           strides=1,
                           l_name=self.l_name + '_block{}'.format(i)))

    def call(self, inputs, training=None, inject=None, inj_args=None):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}

        outputs = inputs
        for i in range(len(self.blocks.layers)):
            outputs, block_inputs, block_kernels, block_outputs = self.blocks.layers[i](outputs, training=training, inject=inject, inj_args=inj_args)
            layer_inputs.update(block_inputs)
            layer_kernels.update(block_kernels)
            layer_outputs.update(block_outputs)

        return outputs, layer_inputs, layer_kernels, layer_outputs



