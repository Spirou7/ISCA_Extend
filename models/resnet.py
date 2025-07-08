import tensorflow as tf
from config import NUM_CLASSES
from models.residual_block import BasicBlocks
from models.inject_layers import InjectConv2D
from models.random_layers import MyDropout

class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_dims, num_classes=NUM_CLASSES, seed=123):
        super(ResNetTypeI, self).__init__()
        self.seed = seed
        self.l_name = 'resnet18'
        self.in_channel = 64
        self.conv1 = InjectConv2D(filters=64,
                                  kernel_size=(7, 7),
                                  strides=2,
                                  padding="same",
                                  seed=seed,
                                  l_name=self.l_name + '_conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(name=self.l_name + '_bn1')
        self.relu1 = tf.keras.layers.ReLU(name=self.l_name + '_relu1')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same",
                                               name=self.l_name + '_pool1')
        self.dropout = MyDropout(0.15, self.seed)

        self.layer1 = BasicBlocks(filters=64,
                                  strides=1,
                                  num_blocks=layer_dims[0],
                                  l_name=self.l_name + '_layer1')
        self.layer2 = BasicBlocks(filters=128,
                                  strides=2,
                                  num_blocks=layer_dims[1],
                                  l_name=self.l_name + '_layer2')
        self.layer3 = BasicBlocks(filters=256,
                                  strides=2,
                                  num_blocks=layer_dims[2],
                                  l_name=self.l_name + '_layer3')
        self.layer4 = BasicBlocks(filters=512,
                                  strides=2,
                                  num_blocks=layer_dims[3],
                                  l_name=self.l_name + '_layer4')

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D(name=self.l_name + '_avgpool')
        self.fc = tf.keras.layers.Dense(units=num_classes,
                                        activation=tf.keras.activations.softmax,
                                        name=self.l_name + '_fc')

    def call(self, inputs, training=None, mask=None, inject=None, inj_args=None):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}
        outputs = {}

        layer_inputs[self.l_name + '_conv1'] = inputs
        x, conv_x = self.conv1(inputs, inject=inject, inj_args=inj_args)
        layer_kernels[self.l_name + '_conv1'] = self.conv1.weights
        layer_outputs[self.l_name + '_conv1'] = conv_x

        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.pool1(x)

        x, l1_inputs, l1_kernels, l1_outputs = self.layer1(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(l1_inputs)
        layer_kernels.update(l1_kernels)
        layer_outputs.update(l1_outputs)

        x, l2_inputs, l2_kernels, l2_outputs = self.layer2(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(l2_inputs)
        layer_kernels.update(l2_kernels)
        layer_outputs.update(l2_outputs)

        outputs['grad_start'] = x

        x, l3_inputs, l3_kernels, l3_outputs = self.layer3(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(l3_inputs)
        layer_kernels.update(l3_kernels)
        layer_outputs.update(l3_outputs)

        x, l4_inputs, l4_kernels, l4_outputs = self.layer4(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(l4_inputs)
        layer_kernels.update(l4_kernels)
        layer_outputs.update(l4_outputs)

        x = self.avgpool(x)
        output = self.fc(x)

        outputs['logits'] = output
        return outputs, layer_inputs, layer_kernels, layer_outputs

def resnet_18(seed, m_name):
    return ResNetTypeI(layer_dims=[2, 2, 2, 2], seed=seed)

def resnet_34():
    return ResNetTypeI(layer_dims=[3, 4, 6, 3])

