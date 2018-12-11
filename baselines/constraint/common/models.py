import tensorflow as tf
import numpy as np
from baselines.common.models import fc, conv, conv_to_fc
import tensorflow.contrib.layers as layers


def augment_network_with_constraint_state(base_network):
    def network(placeholders, **conv_kwargs):
        unscaled_images = placeholders[0]
        constraints = placeholders[1:]

        h_main = layers.flatten(base_network(unscaled_images, **conv_kwargs))
        h_main = fc(h_main, 'fc_hmain', nh=512)
        h_cont = [
            fc(c, 'fc_' + str(i), nh=4) for i, c in enumerate(constraints)
        ]
        return tf.concat([h_main] + h_cont, axis=-1)
    return network