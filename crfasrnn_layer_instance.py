import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
from keras import backend as K

import high_dim_filter_loader
custom_module = high_dim_filter_loader.custom_module

def _diagonal_initializer(shape):
    return np.eye(shape[0], shape[1], dtype=np.float32)

def _potts_model_initializer(shape):
    return -1 * _diagonal_initializer(shape)


def random_initializer(shape):
    return 1

class CrfRnnLayerForInstance(Layer):
   

    def __init__(self, image_dims ,num_detections,num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations, **kwargs):
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.num_detections = num_detections
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        self.box_weight = None
        self.global_weight = None
        super(CrfRnnLayerForInstance, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weights of the spatial kernel
        self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                   shape=(self.num_detections, self.num_detections),
                                                   initializer=_diagonal_initializer,
                                                   trainable=True)

        # Weights of the bilateral kernel
        self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                     shape=(self.num_detections, self.num_detections),
                                                     initializer=_diagonal_initializer,
                                                     trainable=True)

        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_detections, self.num_detections),
                                                    initializer=_potts_model_initializer,
                                                    trainable=True)

        self.box_weight = self.add_weight(name='box_weight',shape=(1,), initializer='uniform', trainable=False)
        
        self.global_weight = self.add_weight(name='global_weight',shape=(1,), initializer='uniform', trainable=False)
        
        super(CrfRnnLayerForInstance, self).build(input_shape)

    def call(self, inputs):
        unaries = (inputs[1][0,:, :, :]*self.box_weight) + (inputs[2][0,:,:,:]*self.global_weight)
        
        rgb = tf.transpose(inputs[0][0,:, :, :], perm=(2, 0, 1))

        d, c, h, w = self.num_detections, self.num_classes, self.image_dims[0], self.image_dims[1]
        
        all_ones = np.ones(( h, w, d), dtype=np.float32)

        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                          theta_gamma=self.theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                            theta_alpha=self.theta_alpha,
                                                            theta_beta=self.theta_beta)
        q_values = unaries

        for i in range(self.num_iterations):
            softmax_out = tf.nn.softmax(q_values, 0)

            # Spatial filtering
            spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                        theta_gamma=self.theta_gamma)
            spatial_out = spatial_out / spatial_norm_vals

            # Bilateral filtering
            bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                          theta_alpha=self.theta_alpha,
                                                          theta_beta=self.theta_beta)
            bilateral_out = bilateral_out / bilateral_norm_vals

            # Weighting filter outputs
            message_passing = (tf.matmul(self.spatial_ker_weights,
                                         tf.reshape(spatial_out, (d, -1))) +
                               tf.matmul(self.bilateral_ker_weights,
                                         tf.reshape(bilateral_out, (d, -1))))

            # Compatibility transform
            pairwise = tf.matmul(self.compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(pairwise, (h, w,d))
            q_values = unaries - pairwise

        return tf.transpose(tf.reshape(q_values, (1, d, h, w)), perm=(0, 2, 3, 1))

    def compute_output_shape(self, input_shape):
        return input_shape