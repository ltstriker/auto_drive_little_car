'''

tfVAERNN.py

a variation autoencode based recurrent neural network (VAE-RNN)
It takes in images array and directly output steering and throttle.

'''

import tensorflow as tf
import numpy as np
import os
import argparse
import json
import time



#from netbase import Netbase
from Aura.parts.config import cfg
#import netbase
from Aura.parts.netbase import Netbase

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

CFG = cfg

#foe debug
import math


class VAE(Netbase):
    def __init__(self, phase, *args, **kwargs):
        super(VAE, self).__init__(*args, **kwargs)
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._run_phase = 'run'
        self._phase = tf.constant(phase, dtype=tf.string)
        self.is_training = tf.equal(self._phase, self._train_phase)
        self.is_running = (phase == self._run_phase)
        self.z_dim = CFG.VAE.Z_DIM
        self.image_shape = CFG.VAE.IMG_CROP_SIZE
        self.g = tf.Graph()
        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.allow_growth = CFG.VAE.TF_ALLOW_GROWTH
        self.sess = tf.Session(graph=self.g, config=self.sess_config)
        if self.is_running:
            with self.g.as_default():
                self.test_x = tf.placeholder(dtype=tf.float32, shape=[None, CFG.VAE.IMG_HEIGHT, CFG.VAE.IMG_WIDTH, 3], name='input')
            self.saver = None

    def preprocess(self, images):
        with tf.variable_scope('resize'):
            x = tf.image.resize_images(images, self.image_shape, tf.image.ResizeMethod.BILINEAR)
            x = tf.divide(x, 255.)
            tf.summary.image('observed image', x, 20)

        return x

    def encode(self, x, batch_size):

        with tf.variable_scope('encoder'):
            h = tf.layers.conv2d(x, 32, 7, strides=2, activation=tf.nn.relu, name="conv1")
            h = tf.layers.batch_normalization(h, training=self.is_training)
            h = tf.layers.conv2d(h, 64, 6, strides=2, activation=tf.nn.relu, name="conv2")
            h = tf.layers.batch_normalization(h, training=self.is_training)
            h = tf.layers.conv2d(h, 128, 7, strides=2, activation=tf.nn.relu, name="conv3")
            h = tf.layers.batch_normalization(h, training=self.is_training)
            h = tf.layers.conv2d(h, 256, 7, strides=2, activation=tf.nn.relu, name="conv4")
            h = tf.layers.batch_normalization(h, training=self.is_training)
            h = tf.reshape(h, [-1, 2 * 2 * 256])

            mu = tf.layers.dense(h, self.z_dim, name="enc_fc_mu")
            logvar = tf.layers.dense(h, self.z_dim, name="enc_fc_log_var")
            sigma = tf.exp(logvar / 2.0)
            epsilon = tf.random_normal([batch_size, self.z_dim])
            z = mu + sigma * epsilon
        return z, mu, logvar

    def decode(self, z):
        with tf.variable_scope('decoder'):
            h = tf.layers.dense(z, 4 * 256, name="dec_fc")
            h = tf.reshape(h, [-1, 1, 1, 4 * 256])
            h = tf.layers.conv2d_transpose(h, 128, 8, strides=2, activation=tf.nn.relu, name="deconv1")
            h = tf.layers.batch_normalization(h, training=self.is_training)
            h = tf.layers.conv2d_transpose(h, 64, 10, strides=2, activation=tf.nn.relu, name="deconv2")
            h = tf.layers.batch_normalization(h, training=self.is_training)
            h = tf.layers.conv2d_transpose(h, 32, 10, strides=2, activation=tf.nn.relu, name="deconv3")
            h = tf.layers.batch_normalization(h, training=self.is_training)
            y = tf.layers.conv2d_transpose(h, 3, 10, strides=2, activation=tf.nn.sigmoid, name="deconv4")

            tf.summary.image('reconstructed_image', y, 20)
        return y

    def compute_loss(self, images, batch_size):
        x = self.preprocess(images)
        z, mu, logvar = self.encode(x, batch_size)
        y = self.decode(z)

        with tf.variable_scope('loss'):
            # reconstruction loss
            r_loss = tf.reduce_sum(tf.square(x - y), reduction_indices=[1, 2, 3])
            r_loss = tf.reduce_mean(r_loss)
            tf.summary.scalar('r_loss', r_loss)

            # augmented kl loss per dim
            kl_loss = - 0.5 * tf.reduce_sum((1 + logvar - tf.square(mu) - tf.exp(logvar)), reduction_indices=1)
            kl_loss = tf.maximum(kl_loss, 0.5 * self.z_dim)
            kl_loss = tf.reduce_mean(kl_loss)
            tf.summary.scalar('kl_loss', kl_loss)

            # loss
            loss = r_loss + kl_loss
            tf.summary.scalar('vae_loss', loss)
        return loss, r_loss, kl_loss

    def train(self, dir, pretrained_model=None, saved_model=None, mode='vae'):
        images = self.load_image_dataset(dir)
        tf.reset_default_graph()
        if mode == 'vae':
            net = VAE(phase='train')
        else:
            raise ValueError("Only VAE model is available! Please set the mode as VAE!")
        with net.g.as_default():
            x = tf.placeholder(dtype=tf.float32, shape=[None, images.shape[1], images.shape[2], 3], name='input')

            loss, r_loss, kl_loss = net.compute_loss(x, CFG.VAE.BATCH_SIZE)

            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(CFG.VAE.LEARNING_RATE, global_step, CFG.VAE.LR_DECAY_STEPS, CFG.VAE.LR_DECAY_RATE, staircase=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)

            saver = tf.train.Saver(max_to_keep=1)
            model_save_dir = CFG.VAE.MODEL_SAVE_DIR
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            train_start_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
            model_name = '{:s}_{:s}.ckpt'.format(mode, str(train_start_time))
            model_save_path = os.path.join(model_save_dir, model_name)

            tboard_save_path = CFG.VAE.TBOARD_SAVE_DIR
            if not os.path.exists(tboard_save_path):
                os.makedirs(tboard_save_path)
            merged = tf.summary.merge_all()

            summary_writer = tf.summary.FileWriter(tboard_save_path)
            summary_writer.add_graph(net.sess.graph)

            train_epochs = CFG.VAE.EPOCHS
            print("train_epochs: ", train_epochs)

            with net.sess.as_default():
                tf.train.write_graph(graph_or_graph_def=net.sess.graph, logdir='', name='{:s}/{:s}.pb'.format(model_save_dir, mode))
                init = tf.global_variables_initializer()
                net.sess.run(init)

                if saved_model is not None:
                    print('Restore model from last model check point{:s}'.format(saved_model))
                    saver.restore(sess=net.sess, save_path=saved_model)

                elif pretrained_model is not None and mode == 'vae_fpl':
                    print("Use pretrained model")
                    net.load_pretrained_weights(pretrained_model, graph, net.sess)

                else:
                    print('Training from scratch')

                for epoch in range(train_epochs):
                    train_steps = len(images) // CFG.VAE.BATCH_SIZE
                    np.random.shuffle(images)
                    output_log = "Epoch #%d" % epoch
                    print(output_log)
                    for step in range(train_steps):
                        train_batch = images[step * CFG.VAE.BATCH_SIZE: (step + 1) * CFG.VAE.BATCH_SIZE]
                        _, train_loss, train_r_loss, train_kl_loss, train_merged, train_step = \
                            net.sess.run([train_op, loss, r_loss, kl_loss, merged, global_step],
                                            feed_dict={x: train_batch})

                        if train_step % 20 == 0:
                            output_log = "step: %d, loss: %.2f, reconstruction_loss: %.2f, KL_loss: %.2f" % \
                                         (train_step, train_loss, train_r_loss, train_kl_loss)
                            print(output_log)
                            summary_writer.add_summary(summary=train_merged, global_step=train_step)
                    saver.save(sess=net.sess, save_path=model_save_path, global_step=epoch)
                    print("Model saved!")
            model= model_save_path + ('-%d' % (train_epochs-1))
            print(model)
            net.sess.close()
            summary_writer.close()
            return model

    def load_pretrained_weights(self, pretrained_model, graph, sess):
        pass

    def generate_series(self, dir, saved_model, series_file, mode='vae'):
        images, labels = self.load_series_dataset(dir)

        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            x = tf.placeholder(dtype=tf.float32, shape=[None, images.shape[1], images.shape[2], 3], name='input')
            phase = 'test'
            if mode == 'vae':
                net = VAE(phase=phase)
            else:
                raise ValueError("Only VAE model is available! Please set the mode as VAE!")

            batch_size = CFG.RNN.MAX_SEQ + 1
            x_preprocess = net.preprocess(x)
            z, mu, logvar = net.encode(x_preprocess, batch_size)
            saver = tf.train.Saver()
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = CFG.VAE.TF_ALLOW_GROWTH

            sess = tf.Session(config=sess_config)

            with sess.as_default():
                print('Restore model from last model point{:s}'.format(saved_model))
                saver.restore(sess=sess, save_path=saved_model)
                num_batch = len(images) // batch_size
                rnn_input = []
                rnn_label = []
                for step in range(num_batch):
                    image_batch = images[step * batch_size: (step + 1) * batch_size]
                    label_batch = labels[step * batch_size: (step + 1) * batch_size]
                    z_batch = sess.run(z, feed_dict={x: image_batch})
                    # This operation zip the image (input) with the previous label (output); this is for RNN usage
                    conc = [np.concatenate([x, y]) for x, y in zip(z_batch[1:], label_batch[:-1])]
                    rnn_input.append(np.array(conc))
                    rnn_label.append(np.array(label_batch[1:]))
            sess.close()

        rnn_input = np.array(rnn_input)
        print("Input shape:", rnn_input.shape)
        rnn_label = np.array(rnn_label)
        print("Label shape:", rnn_label.shape)
        print("Save series file...")
        series_save_dir = CFG.VAE.SERIES_SAVE_DIR
        if not os.path.exists(series_save_dir):
            os.makedirs(series_save_dir)
        np.savez_compressed(os.path.join(series_save_dir, series_file), input=rnn_input, label=rnn_label)
        series = os.path.join(series_save_dir,series_file)
        print(series)
        return series

    def generate_batch(self, dir, saved_model, train_series_file, val_series_file):
        images, labels = self.load_series_dataset(dir)
        images = np.array(images)
        labels = np.array(labels)
        batch_size = CFG.CNN.BATCH_SIZE
        total_data = len(images)
        total_train = int(total_data * CFG.CNN.TRAIN_VAL_SPLIT)
        print(total_train)
        total_train -= total_train % batch_size
        total_val = total_data - total_train
        total_val -= total_val % batch_size
        train_images = images[:total_train]
        val_images = images[total_train:]
        train_label = labels[:total_train]
        val_label = labels[total_train:]
        train_step = len(train_images) // batch_size
        val_step = len(val_images) // batch_size
        tf.reset_default_graph()
        graph = tf.Graph()
        print('Restore model from vae: {:s}'.format(saved_model))


        with graph.as_default():
            x = tf.placeholder(dtype=tf.float32, shape=[None, images.shape[1], images.shape[2], 3], name='input')
            net = VAE(phase='test')
            x_preprocess = net.preprocess(x)
            z, mu, logvar = net.encode(x_preprocess, batch_size=batch_size)
            saver = tf.train.Saver()
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = CFG.VAE.TF_ALLOW_GROWTH
            sess = tf.Session(config=sess_config)

            with sess.as_default():
                saver.restore(sess=sess, save_path=saved_model)

                cnn_train_input = []
                cnn_train_label = []
                cnn_val_input = []
                cnn_val_label = []

                train_index = np.random.permutation(total_train)
                val_index = np.random.permutation(total_val)
                train_images_shuffle = train_images[train_index].reshape(-1, batch_size, CFG.VAE.IMG_HEIGHT, CFG.VAE.IMG_WIDTH, 3)
                val_images_shuffle = val_images[val_index].reshape(-1, batch_size, CFG.VAE.IMG_HEIGHT, CFG.VAE.IMG_WIDTH, 3)
                train_label_shuffle = train_label[train_index].reshape(-1, batch_size, 2)
                val_label_shuffle = val_label[val_index].reshape(-1, batch_size, 2)

                for step in range(train_step):
                    train_images_batch = train_images_shuffle[step]
                    train_input_batch = sess.run(z, feed_dict={x: train_images_batch})
                    train_label_batch = train_label_shuffle[step]
                    cnn_train_input.append(train_input_batch)
                    cnn_train_label.append(train_label_batch)

                for step in range(val_step):
                    val_images_batch = val_images_shuffle[step]
                    val_input_batch = sess.run(z, feed_dict={x: val_images_batch})
                    val_label_batch = val_label_shuffle[step]
                    cnn_val_input.append(val_input_batch)
                    cnn_val_label.append(val_label_batch)

            cnn_train_input = np.array(cnn_train_input)
            print("Train input shape:", cnn_train_input.shape)
            cnn_train_label = np.array(cnn_train_label)
            print("Train label shape:", cnn_train_label.shape)
            print("Save train series file...")
            series_save_dir = CFG.VAE.SERIES_SAVE_DIR
            if not os.path.exists(series_save_dir):
                os.makedirs(series_save_dir)
            np.savez_compressed(os.path.join(series_save_dir, train_series_file), input=cnn_train_input, label=cnn_train_label)
            train_series = os.path.join(series_save_dir, train_series_file)
            print("Saved to " + train_series)

            cnn_val_input = np.array(cnn_val_input)
            print("Train input shape:", cnn_val_input.shape)
            cnn_val_label = np.array(cnn_val_label)
            print("Train label shape:", cnn_val_label.shape)
            print("Save validation series file...")
            series_save_dir = CFG.VAE.SERIES_SAVE_DIR
            if not os.path.exists(series_save_dir):
                os.makedirs(series_save_dir)
            np.savez_compressed(os.path.join(series_save_dir, val_series_file), input=cnn_val_input, label=cnn_val_label)
            val_series = os.path.join(series_save_dir, val_series_file)
            print("Saved to " + val_series)

            return train_series, val_series


class CNN(Netbase):
    def __init__(self, phase, *args, **kwargs):
        super(CNN, self).__init__(*args, **kwargs)
        self._train_phase = 'train'
        self._run_phase = 'run'
        self.is_training = (phase == self._train_phase)
        self.is_running = (phase == self._run_phase)
        self.input_size = [None, CFG.VAE.Z_DIM]
        self.ANGLE_DIM = [None, 15]  # one_hot
        self.THROTTLE_DIM = [None, 1]
        self.g = tf.Graph()
        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.allow_growth = CFG.VAE.TF_ALLOW_GROWTH
        self.sess = tf.Session(graph=self.g, config=self.sess_config)
        self.x = None
        self.angle_weight = CFG.CNN.ANGLE_WEIGHT
        self.throttle_weight = CFG.CNN.THROTTLE_WEIGHT
        self.learning_rate = CFG.CNN.LEARNING_RATE
        if self.is_running:
            with self.g.as_default():
                self.saver = None
        if self.is_training:
            self.angle_out = None
            self.throttle_out = None
            self.angle_target = None
            self.throttle_target = None
            self.angle_loss = None
            self.throttle_loss = None

    def linear_bin(self, a):
        a = a + 1
        b = round(a / (2 / 14))
        arr = np.zeros(15)
        arr[int(b)] = 1
        return arr

    def set_model_params(self, params):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                pshape = self.sess.run(var).shape
                p = np.array(params[idx])
                assert pshape == p.shape, "inconsistent shape"
                assign_op = var.assign(p.astype(np.float)/10000.)
                self.sess.run(assign_op)
                idx += 1

    def load_json(self, jsonfile='model.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)

    def build_graph(self, x):
        self.x = x
        z = tf.layers.dense(self.x, 64, activation=tf.nn.relu, name='layer1')
        z = tf.nn.dropout(z, 0.9, name="dropout1")
        z = tf.layers.dense(z, 32, activation=tf.nn.relu, name='layer2')
        z = tf.nn.dropout(z, 0.9, name="dropout2")
        self.angle_out = tf.layers.dense(z, 15, activation=tf.nn.softmax, name="angle_out")  # category probability 15
        self.throttle_out = tf.layers.dense(z, 1, activation=None, name="throttle_out")

        return self.angle_out, self.throttle_out

    def train(self, saved_model, train_file_dir, val_file_dir, epochs=100, exit_k=5):
        print("Start loading training data...")
        train_input, train_label = self.load_rnn_dataset(train_file_dir)
        print("Start loading val data...")
        val_input, val_label = self.load_rnn_dataset(val_file_dir)
        train_steps = len(train_input)
        val_steps = len(val_input)

        with self.g.as_default():
            x = tf.placeholder(tf.float32, shape=self.input_size, name='input')
            self.angle_out, self.throttle_out = self.build_graph(x)

            self.angle_target = tf.placeholder(tf.float32, shape=self.ANGLE_DIM, name='angle_target')
            self.throttle_target = tf.placeholder(tf.float32, shape=self.THROTTLE_DIM, name='throttle_target')

            self.angle_loss = -tf.reduce_sum(self.angle_target * tf.log(tf.clip_by_value(self.angle_out, 1e-7, 1 - 1e-7)), axis=-1)
            self.angle_loss = tf.reduce_mean(self.angle_loss)

            self.throttle_loss = tf.reduce_mean(tf.abs(self.throttle_target - self.throttle_out), axis=-1)
            self.throttle_loss = tf.reduce_mean(self.throttle_loss)

            angle_weight = tf.Variable(self.angle_weight, trainable=False, name="angle_weight")
            throttle_weight = tf.Variable(self.throttle_weight, trainable=False, name="throttle_weight")
            self.loss = tf.add(tf.multiply(angle_weight, self.angle_loss), tf.multiply(throttle_weight, self.throttle_loss))

            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.lr = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, 0.96, staircase=True)  ####### not sure if we shall add learning rate

            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            init = tf.global_variables_initializer()
            with self.sess.as_default():
                self.sess.run(init)

                val_loss_min = 10000
                earlystop_num = 0
                earlystop_flag = 0

                for epoch in range(epochs):
                    for train_step in range(train_steps):
                        img_batch = train_input[train_step]
                        label_batch = train_label[train_step]
                        angle_batch = label_batch[:, 1]
                        angle_batch = [self.linear_bin(y) for y in angle_batch]
                        throttle_batch = label_batch[:, 0].reshape(-1, 1)
                        feed = {x: img_batch, self.angle_target: angle_batch, self.throttle_target: throttle_batch}
                        loss, angle_loss, throttle_loss, step, _ = self.sess.run(
                            [self.loss, self.angle_loss, self.throttle_loss, self.global_step, self.train_op], feed)
                        if (step + 1) % 10 == 0:
                            output_log = "step: %d, loss: %.6f, angle_loss: %.6f, throttle_loss: %.6f" % (
                            (step + 1), loss, angle_loss, throttle_loss)
                            print(output_log)

                    val_loss = val_angle_loss = val_throttle_loss = 0
                    for val_step in range(val_steps):
                        img_batch = val_input[val_step]
                        label_batch = val_label[val_step]
                        angle_batch = label_batch[:, 1]
                        angle_batch = [self.linear_bin(y) for y in angle_batch]
                        throttle_batch = label_batch[:, 0].reshape(-1, 1)
                        val_feed = {self.x: img_batch, self.angle_target: angle_batch, self.throttle_target: throttle_batch}
                        loss, angle_loss, throttle_loss = self.sess.run([self.loss, self.angle_loss, self.throttle_loss],
                                                                        val_feed)
                        val_loss += loss
                        val_throttle_loss += throttle_loss
                        val_angle_loss += angle_loss

                    val_loss /= val_steps
                    val_angle_loss /= val_steps
                    val_throttle_loss /= val_steps

                    output_log = "epoch: %d, val_loss: %.6f, val_angle_loss: %.6f, val_throttle_loss: %.6f" % (
                    epoch, val_loss, val_angle_loss, val_throttle_loss)
                    print("\n")
                    print(output_log)
                    print("\n")

                    if val_loss >= val_loss_min:
                        earlystop_num = earlystop_num + 1
                    else:
                        earlystop_num = 0
                        self.save_json(saved_model)
                        print("Saved Model")
                        val_loss_min = val_loss

                    if earlystop_num >= exit_k:
                        earlystop_flag = 1

                    if earlystop_flag:
                        print("Early Stop")
                        break


class RNN(Netbase):
    def __init__(self, phase, layer_norm, *args, **kwargs):
        super(RNN, self).__init__(*args, **kwargs)
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._run_phase = 'run'
        self._phase = tf.constant(phase, dtype=tf.string)
        self._layer_norm = layer_norm
        self.is_training = tf.equal(self._phase, self._train_phase)
        self.is_testing = tf.equal(self._phase, self._test_phase)
        self.is_running = (phase == self._run_phase)
        self.hidden_size = 256  # hidden unit for LSTM
        self.outwidth = CFG.RNN.OUTWIDTH
        self.train_output = None
        self.train_last_state = None
        self.g = tf.Graph()
        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.allow_growth = CFG.RNN.TF_ALLOW_GROWTH
        self.sess = tf.Session(graph=self.g, config=self.sess_config)

        if self.is_running:
            self.run_x = None
            self.run_l = None
            self.initial_state = None
            self.saver = None
            with self.g.as_default():
                self.run_x = tf.placeholder(dtype=tf.float32, shape=[None, CFG.RNN.VAL_MAX_SEQ, CFG.RNN.INWIDTH],
                                            name='input')
                self.run_l = tf.placeholder(dtype=tf.float32, shape=[None, CFG.RNN.VAL_MAX_SEQ, CFG.RNN.OUTWIDTH],
                                            name='output')

    def build_model(self, x, batch_size):
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size, layer_norm=self._layer_norm)
        self.initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        output, last_state = tf.nn.dynamic_rnn(cell, x, initial_state=self.initial_state, time_major=False,
                                               swap_memory=True, dtype=tf.float32, scope="RNN")

        output = tf.reshape(output, [-1, self.hidden_size])
        with tf.variable_scope('LinearRegression'):
            output_w = tf.get_variable("output_w", [self.hidden_size, self.outwidth])
            output_b = tf.get_variable("output_b", [self.outwidth])
            output = tf.nn.xw_plus_b(output, output_w, output_b)

        return output, last_state

    def compute_loss(self, x, label, batch_size):
        self.train_output, self.train_last_state = self.build_model(x, batch_size)
        label = tf.reshape(label, [-1, self.outwidth])

        throttle_predict, angle_predict = tf.split(self.train_output, 2, 1)
        throttle_label, angle_label = tf.split(label, 2, 1)

        angle_loss = tf.reduce_mean((angle_predict - angle_label) ** 2)
        tf.summary.scalar('angle loss', angle_loss)
        throttle_loss = tf.reduce_mean((throttle_predict - throttle_label) ** 2)
        tf.summary.scalar('throttle loss', throttle_loss)
        loss = angle_loss + throttle_loss
        tf.summary.scalar('rnn_loss', loss)

        return loss, angle_loss, throttle_loss

    def train(self, file, saved_model=None):
        input, label = self.load_rnn_dataset(file)
        tf.reset_default_graph()
        net = RNN('train', CFG.RNN.LAYER_NORM)
        with net.g.as_default():
            x = tf.placeholder(dtype=tf.float32, shape=[None, CFG.RNN.MAX_SEQ, CFG.RNN.INWIDTH], name='input')
            l = tf.placeholder(dtype=tf.float32, shape=[None, CFG.RNN.MAX_SEQ, CFG.RNN.OUTWIDTH], name='input')

            loss, angle_loss, throttle_loss = net.compute_loss(x, l, CFG.RNN.BATCH_SIZE)

            with tf.variable_scope('training'):
                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(CFG.RNN.LEARNING_RATE, global_step, CFG.RNN.LR_DECAY_STEPS,
                                                           CFG.RNN.LR_DECAY_RATE, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                gvs = optimizer.compute_gradients(loss)
                capped_gvs = [(tf.clip_by_value(grad, -CFG.RNN.GRAD_CLIP, CFG.RNN.GRAD_CLIP), var) for grad, var in gvs]
                train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            saver = tf.train.Saver(max_to_keep=1)
            model_save_dir = CFG.RNN.MODEL_SAVE_DIR
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            train_start_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
            model_name = 'rnn_{:s}.ckpt'.format(str(train_start_time))
            model_save_path = os.path.join(model_save_dir, model_name)

            tboard_save_path = CFG.RNN.TBOARD_SAVE_DIR
            if not os.path.exists(tboard_save_path):
                os.makedirs(tboard_save_path)
            merged = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(tboard_save_path)
            summary_writer.add_graph(net.sess.graph)

            train_epochs = CFG.RNN.EPOCHS
            print("train_epochs: ", train_epochs)

            with net.sess.as_default():
                tf.train.write_graph(graph_or_graph_def=net.sess.graph, logdir='',
                                     name='{:s}/rnn.pb'.format(model_save_dir))
                init = tf.global_variables_initializer()
                net.sess.run(init)

                if saved_model is not None:
                    print('Restore model from last model check point{:s}'.format(saved_model))
                    saver.restore(sess=net.sess, save_path=saved_model)

                else:
                    print('Training from scratch')

                num_data = len(input)
                train_steps = num_data // CFG.RNN.BATCH_SIZE
                print("train steps per epoch: ", train_steps)

                for epoch in range(train_epochs):
                    # training part
                    for step in range(train_steps):
                        input_batch = input[step*CFG.RNN.BATCH_SIZE:(step+1)*CFG.RNN.BATCH_SIZE]
                        label_batch = label[step*CFG.RNN.BATCH_SIZE:(step+1)*CFG.RNN.BATCH_SIZE]
                        feed = {x: input_batch, l: label_batch}
                        train_loss, train_throttle_loss, train_angle_loss, train_step, train_merged, _ = net.sess.run(
                            [loss, throttle_loss, angle_loss, global_step, merged, train_op], feed)
                        summary_writer.add_summary(train_merged, global_step=epoch)
                        if train_step % 9 == 0:
                            output_log = "step: %d, loss: %.6f, angle_loss: %.6f, throttle_loss: %.6f" % (
                            train_step + 1, train_loss, train_angle_loss, train_throttle_loss)
                            print(output_log)
                            saver.save(sess=net.sess, save_path=model_save_path, global_step=epoch)
            net.sess.close()
            summary_writer.close()


class VAECNN(object):
    def __init__(self, phase, vae_model=None, cnn_model=None):
        self.phase = phase
        self.rnn_layer_norm = CFG.RNN.LAYER_NORM
        self.vae_model = VAE(self.phase)
        self.cnn_model = CNN(phase=self.phase)
        if self.phase == 'run':
            with self.vae_model.g.as_default():
                x_preprocess = self.vae_model.preprocess(self.vae_model.test_x)
                self.z, self.mu, self.logvar = self.vae_model.encode(x_preprocess, batch_size=1)
                self.vae_model.saver = tf.train.Saver()
                with self.vae_model.sess.as_default():
                    self.vae_model.saver.restore(sess=self.vae_model.sess, save_path=vae_model)

            with self.cnn_model.g.as_default():
                self.cnn_input = tf.placeholder(tf.float32, shape=self.cnn_model.input_size, name='input')
                self.angle_out, self.throttle_out = self.cnn_model.build_graph(x=self.cnn_input)
                init = tf.global_variables_initializer()
                self.cnn_model.sess.run(init)
                self.cnn_model.load_json(cnn_model)


    def linear_unbin(self, arr):
        if not len(arr) == 15:
            raise ValueError('Illegal array length, must be 15')
        b = np.argmax(arr)
        a = b * (2 / 14) - 1
        return a

    def train(self, data_dir, train_series_file, val_series_file, cnn_saved_model):
        vae_model = self.vae_model.train(dir=data_dir, pretrained_model=None, saved_model=None, mode='vae')
        train_series, val_series = self.vae_model.generate_batch(data_dir, vae_model, train_series_file, val_series_file)
        self.cnn_model.train(cnn_saved_model, train_series, val_series)

    def run(self, img):
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        hidden_representation = self.vae_model.sess.run(self.z, feed_dict={self.vae_model.test_x: img})
        angle_out, throttle_out = self.cnn_model.sess.run([self.angle_out, self.throttle_out], feed_dict={self.cnn_input: hidden_representation})
        angle_out = self.linear_unbin(angle_out[0])

        #for debug
        #print(angle_out,math.tanh(angle_out*2.5))
        #angle_out=math.tanh(angle_out*2.5)

        return throttle_out[0][0],angle_out

    def main(self, args):
        print("Start VAECNN for " + self.phase)
        if self.phase == 'train':
            data_dir = args.datadir
            series_file = args.seriesfile
            self.train(data_dir, series_file)
        if self.phase == 'run':
            img_dir = args.imgdir
            netbase = Netbase()
            images, labels = netbase.load_series_dataset(img_dir)
            acc_test_loss = 0
            for image, label in zip(images, labels):
                output = self.run(image)
                angle_output = output[0]
                throttle_output = output[1]
                throttle_test_loss = abs(throttle_output - label[0])
                angle_test_loss = abs(angle_output - label[1])
                test_loss = throttle_test_loss * CFG.CNN.THROTTLE_WEIGHT + angle_test_loss * CFG.CNN.ANGLE_WEIGHT
                acc_test_loss += test_loss
                print("output: ", output)
                print('loss: %.6f, throttle_loss: %.6f, angle_loss: %.6f' % (test_loss, throttle_test_loss, angle_test_loss))
                print("###############")
            acc_test_loss /= len(images)
            print('acc_test_loss: %.6f' % acc_test_loss)


class VAERNN(object):
    def __init__(self, phase, vae_model=None, rnn_model=None):
        self.phase = phase
        self.rnn_layer_norm = CFG.RNN.LAYER_NORM
        self.vae_model = VAE(self.phase)
        self.cnn_model = CNN(self.phase)
        self.rnn_model = RNN(phase=self.phase, layer_norm=self.rnn_layer_norm)
        if self.phase == 'run':
            self.image = None
            self.reconstruct = None
            self.prev_label = np.array([0, 0])
            self.prev_state = None

            with self.vae_model.g.as_default():
                x_preprocess = self.vae_model.preprocess(self.vae_model.test_x)
                self.z, self.mu, self.logvar = self.vae_model.encode(x_preprocess, batch_size=1)
                self.vae_model.saver = tf.train.Saver()
                with self.vae_model.sess.as_default():
                    self.vae_model.saver.restore(sess=self.vae_model.sess, save_path=vae_model)


            with self.rnn_model.g.as_default():
                self.output, self.next_label = self.rnn_model.build_model(x=self.rnn_model.run_x, batch_size=1)
                self.rnn_model.saver = tf.train.Saver()
                with self.rnn_model.sess.as_default():
                    self.rnn_model.saver.restore(sess=self.rnn_model.sess, save_path=rnn_model)
                    self.prev_state = self.rnn_model.sess.run(self.rnn_model.initial_state)


    def train(self, data_dir, series_file):
        # Train VAE
        vae_model = self.vae_model.train(dir=data_dir, pretrained_model=None, saved_model=None, mode='vae')
        # Generate the series files for RNN
        series = self.vae_model.generate_series(dir=data_dir, saved_model=vae_model, series_file=series_file, mode='vae')
        # Train RNN
        self.rnn_model.train(file=series, saved_model=None)

    def run(self, img):
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        hidden_representation = self.vae_model.sess.run(self.z, feed_dict={self.vae_model.test_x: img})
        hidden_representation = hidden_representation.reshape((CFG.RNN.VAL_BATCH_SIZE, CFG.RNN.VAL_MAX_SEQ,
                                                               hidden_representation.shape[1]))
        prev_label = self.prev_label.reshape((CFG.RNN.VAL_BATCH_SIZE, CFG.RNN.VAL_MAX_SEQ, CFG.RNN.OUTWIDTH))
        rnn_input = np.concatenate([hidden_representation, prev_label], axis=2)
        output, next_state = self.rnn_model.sess.run([self.output, self.next_label], feed_dict={self.rnn_model.run_x:
                                                                                                rnn_input, self.rnn_model.initial_state: self.prev_state})
        self.prev_state = next_state
        self.prev_label = output[0]
        return output

    def main(self, args):
        print("Start VAERNN...")
        print(self.phase)
        if self.phase == 'train':
            data_dir = args.datadir
            series_file = args.seriesfile
            self.train(data_dir, series_file)
        if self.phase == 'run':
            img_dir = args.img
            netbase = Netbase()
            images, label = netbase.load_series_dataset(img_dir)
            # prev_label = np.array([1, 1])
            acc_test_loss = 0
            VAERNN.prev_label = label[0]
            i = 0

            for image in images[1:]:
                i += 1
                output = self.run(image)
                throttle_output = output[0][0]
                angle_output = output[0][1]
                throttle_test_loss = abs(throttle_output - label[i][0])
                angle_test_loss = abs(angle_output - label[i][1])
                test_loss = throttle_test_loss + angle_test_loss
                acc_test_loss = (acc_test_loss * (i-1) + throttle_test_loss + angle_test_loss)/(i)
                self.prev_label = label[i]
                print("output: ", output)
                log = 'acc_loss: %.6f, loss: %.6f, throttle_loss: %.6f, angle_loss: %.6f' % (acc_test_loss, test_loss, throttle_test_loss, angle_test_loss)
                print(log)
                print("###############")

            self.vae_model.sess.close()
            self.rnn_model.sess.close()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='VAERNN')
    # parser.add_argument('--mode', type=str, help='train/run VAERNN/VAECNN')
    # parser.add_argument('--datadir', type=str, help='data for training')
    # parser.add_argument('--seriesfile', type=str, default='series.npz', help='series file name')
    # parser.add_argument('--img', type=str, help='images directory')
    # parser.add_argument('--vaemodel', type=str, default=None, help='vae model path')
    # parser.add_argument('--rnnmodel', type=str, default=None, help='rnn model path')
    # args = parser.parse_args()
    # tf.reset_default_graph()
    # mode = args.mode
    # datadir = args.datadir
    # seriesfile = args.seriesfile
    # vae_model = args.vaemodel
    # rnn_model = args.rnnmodel
    # vaernn = VAERNN(phase=mode, vae_model=vae_model, rnn_model=rnn_model)
    # vaernn.main(args)

    parser = argparse.ArgumentParser(description='VAECNN')
    parser.add_argument('--mode', type=str, help='train/run VAECNN')
    parser.add_argument('--datadir', type=str, help='data for training')
    parser.add_argument('--trainseriesfile', type=str, default='trainseries.npz', help='train series file name')
    parser.add_argument('--valseriesfile', type=str, default='valseries.npz', help='val series file name')
    parser.add_argument('--cnnsavemodel', type=str, default=None, help='cnn save model path')
    parser.add_argument('--vaemodel', type=str, default=None, help='vae model for testing')
    parser.add_argument('--cnnmodel', type=str, default=None, help='cnn model for testing')
    parser.add_argument('--imgdir', type=str, default=None, help='images for testing')
    args = parser.parse_args()

    tf.reset_default_graph()
    mode = args.mode
    datadir = args.datadir
    trainseriesfile = args.trainseriesfile
    valseriesfile = args.valseriesfile
    cnnsavemodel = args.cnnsavemodel
    cnnmodel = args.cnnmodel
    vaemodel = args.vaemodel

    if mode == 'train':
        vaecnn = VAECNN(phase=mode)
        vaecnn.train(datadir, trainseriesfile, valseriesfile, cnnsavemodel)
    if mode == 'run':
        vaecnn = VAECNN(phase=mode, vae_model=vaemodel, cnn_model=cnnmodel)
        vaecnn.main(args)



#--mode run --img /home/likewise-open/SENSETIME/wuhongtao/VAE_RNN/data/tub_7_18-09-05 --vaemodel /home/likewise-open/SENSETIME/wuhongtao/aura/nets/model/vae_2018-09-26-16-57.ckpt-9 --rnnmodel /home/likewise-open/SENSETIME/wuhongtao/aura/nets/model/rnn_2018-09-26-17-01.ckpt-199
# --mode train --datadir /home/likewise-open/SENSETIME/wuhongtao/senserover_data/haixiang8f/tub_7_18-10-18 --seriesfile 1022test1.npz
# --mode run --vaemodel /home/likewise-open/SENSETIME/wuhongtao/aura/nets/model/vae_2018-10-22-16-07.ckpt-29 --rnn /home/likewise-open/SENSETIME/wuhongtao/aura/nets/model/rnn_2018-10-22-16-11.ckpt-489 --img /home/likewise-open/SENSETIME/wuhongtao/senserover_data/haixiang8f/tub_7_18-10-18/
