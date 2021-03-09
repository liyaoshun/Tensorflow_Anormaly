# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(lys)s
"""
from __future__ import absolute_import

from keras import backend as K
from keras.engine.topology import Layer
import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages/keras/")
from utils import conv_utils

class _Pooling3D(Layer):
    """Abstract class for different pooling 3D layers.
    """

    def __init__(self, pool_size=(2, 2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        super(_Pooling3D, self).__init__(**kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 3, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=5)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            len_dim1 = input_shape[2]
            len_dim2 = input_shape[3]
            len_dim3 = input_shape[4]
        elif self.data_format == 'channels_last':
            len_dim1 = input_shape[1]
            len_dim2 = input_shape[2]
            len_dim3 = input_shape[3]
        len_dim1 = conv_utils.conv_output_length(len_dim1, self.pool_size[0],
                                                 self.padding, self.strides[0])
        len_dim2 = conv_utils.conv_output_length(len_dim2, self.pool_size[1],
                                                 self.padding, self.strides[1])
        len_dim3 = conv_utils.conv_output_length(len_dim3, self.pool_size[2],
                                                 self.padding, self.strides[2])
        if self.data_format == 'channels_first':
            return (input_shape[0],
                    input_shape[1],
                    len_dim1, len_dim2, len_dim3)
        elif self.data_format == 'channels_last':
            return (input_shape[0],
                    len_dim1, len_dim2, len_dim3,
                    input_shape[4])

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        raise NotImplementedError

    def call(self, inputs):
        output = self._pooling_function(inputs=inputs,
                                        pool_size=self.pool_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        data_format=self.data_format)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'padding': self.padding,
                  'strides': self.strides,
                  'data_format': self.data_format}
        base_config = super(_Pooling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class MaxPooling3D(_Pooling3D):
    """Max pooling operation for 3D data (spatial or spatio-temporal).

    # Arguments
        pool_size: tuple of 3 integers,
            factors by which to downscale (dim1, dim2, dim3).
            (2, 2, 2) will halve the size of the 3D input in each dimension.
        strides: tuple of 3 integers, or None. Strides values.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `channels_first` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Input shape
        - If `data_format='channels_last'`:
            5D tensor with shape:
            `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
        - If `data_format='channels_first'`:
            5D tensor with shape:
            `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    # Output shape
        - If `data_format='channels_last'`:
            5D tensor with shape:
            `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
        - If `data_format='channels_first'`:
            5D tensor with shape:
            `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
    """


    def __init__(self, pool_size=(2, 2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        super(MaxPooling3D, self).__init__(pool_size, strides, padding,
                                           data_format, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        output = K.pool3d(inputs, pool_size, strides,
                          padding, data_format, pool_mode='max')
if __name__ == "__main__":
    print("main")