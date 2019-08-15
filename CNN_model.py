import tensorflow as tf

class ConvModel(object):
    def __init__(self, data_params):
        self.inputs = tf.placeholder(tf.float32, [None,
                                                  data_params.image_height,
                                                  data_params.image_width,
                                                  data_params.image_channel])
        self.labels = tf.placeholder(tf.int64, [None, data_params.char_length])
        
        self.batch_size = data_params.batch_size
        self.num_char_class = data_params.num_char_class
        self.char_length = data_params.char_length
        
    def build_model(self):
        self.cnn_layer()
        self.sequence_fc_layer()
        self.calculate_loss()
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.001,
                                                   momentum=0.9)
        self.train_op = self.optimizer.minimize(self.loss)
        
        self.create_accuracy()
        
    def cnn_layer(self):
        def conv_bn_unit(x, name, filters, strides):
            with tf.variable_scope(name):
                x = tf.layers.conv2d(x,
                                     filters=filters,
                                     kernel_size=3,
                                     strides=strides,
                                     padding='same',
                                     kernel_initializer=tf.glorot_normal_initializer(),
                                     bias_initializer=tf.constant_initializer())
                x = tf.contrib.layers.batch_norm(x,
                                                 decay=0.9,
                                                 center=True,
                                                 scale=True,
                                                 epsilon=1e-5,
                                                 updates_collections=None,
                                                 fused=True,
                                                 data_format='NHWC',
                                                 zero_debias_moving_mean=True,
                                                 scope='BatchNorm')
                x = tf.nn.relu(features=x)
                return x
        
        with tf.variable_scope('cnn_layers'):
            conv1 = conv_bn_unit(self.inputs, 'conv1', 32, 2)
            conv2 = conv_bn_unit(conv1, 'conv2', 32, 1)
            conv3 = conv_bn_unit(conv2, 'conv3', 64, 1)
            pool1 = tf.layers.max_pooling2d(conv3, 2, 2, padding='same')
            conv4 = conv_bn_unit(pool1, 'conv4', 80, 1)
            conv5 = conv_bn_unit(conv4, 'conv5', 192, 2)
            out_layer = conv_bn_unit(conv5, 'conv6', 288, 1)
        self.cnn_feats = out_layer
        
    def sequence_fc_layer(self):
        def fc_unit(x, name):
            with tf.variable_scope(name):
                x = tf.layers.dense(x, 512)
                x = tf.layers.dense(x, self.num_char_class)
            return x
        
        with tf.variable_scope('separate_dense_layer'):
            fc_seq = []
            in_layer = tf.layers.flatten(self.cnn_feats)
            for i in range(self.char_length):
                sub_name = 'sub_fc_idx' + str(i)
                fc_seq.append(fc_unit(in_layer, sub_name))
        self.fc_seq = fc_seq
        
    def calculate_loss(self):
        self.logits = tf.concat([tf.expand_dims(t, 1) for t in self.fc_seq], 1)
        distrib_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                   logits=self.logits)
        self.loss = tf.reduce_mean(distrib_loss, axis=0)
        
        for i in range(self.char_length):
            tf.summary.scalar('loss_char_{0}'.format(i), self.loss[i])
        
    def create_accuracy(self):
        self.preds = tf.argmax(self.logits, 2)
        correct_pred = tf.equal(self.labels, self.preds)
        self.char_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        tf.summary.scalar('char_accuracy', self.char_acc)
        
        correct_count = tf.reduce_sum(tf.cast(correct_pred, tf.float32), axis=1)
        correct_check = tf.cast(correct_count, tf.int64) / 5
        self.str_acc = tf.reduce_mean(tf.cast(correct_check, tf.float32))
        
        tf.summary.scalar('string_accuracy', self.str_acc)