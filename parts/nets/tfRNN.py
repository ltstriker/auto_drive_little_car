# '''

# pilots.py

# Methods to create, use, save and load pilots. Pilots
# contain the highlevel logic used to determine the angle
# and throttle of a vehicle. Pilots can include one or more
# models to help direct the vehicles motion.

# '''

import tensorflow as tf
import numpy as np
import json
from parts.tools import data
from core.config import load_config
cfg=load_config()

sess = None
def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

class TensorflowPilot:
    def shutdown(self):
        pass

    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                param_name = var.name
                p = self.sess.run(var)
                model_names.append(param_name)
                params = np.round(p*10000).astype(np.int).tolist()
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names

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

    def save_json(self, jsonfile='model.json'):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

    def load(self, model_path):
        try:
            self.load_json(model_path)
        except:
            print("Could not restore model")

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()


class CNN(TensorflowPilot):
    def __init__(self, is_training=True, learning_rate=0.001, *args, **kwargs):
        super(CNN, self).__init__(*args, **kwargs)
        self.IMAGE_DIM = [None, cfg['CNN']['CNN_IMG_HEIGHT'],cfg['CNN']['CNN_IMG_WIDTH'] , 3]
        self.ANGLE_DIM = [None, 15]  # one_hot
        self.THROTTLE_DIM = [None, 1]

        self.is_training = is_training
        self.learning_rate = learning_rate

        self.viwer_op=[]

        with tf.variable_scope('Categorical'):
            self.g = tf.Graph()                                     #创建一个新的Graph        g
            with self.g.as_default():                               #将g设为当前默认的graph
                self.build_graph()                                  #在g中建立网络
                init = tf.global_variables_initializer()
                # self.saver = tf.train.Saver(max_to_keep=1)
        self.sess = tf.Session(graph=self.g)                        #创建一个连接到g的Session  sess
        self.sess.run(init)                                         #初始化g的网络

    def Viwer(self,chs,layers):
        range_stop = chs // 3
        size_splits = [3 for i in range(0, range_stop)]
        if len(size_splits) * 3 < chs:
            size_splits.append(chs % 3)
        conv1_split = tf.split(layers, num_or_size_splits=size_splits, axis=3)  # conv1.shape = [128,24,24,64]

        conv1_concats_1 = []
        concat_step = len(conv1_split) // 2
        for i in range(0, concat_step, 2):
            concat = tf.concat([conv1_split[i], conv1_split[i + 1]], axis=1)
            conv1_concats_1.append(concat)

        conv1_concats_2 = []
        concat_step = len(conv1_concats_1) // 2
        for i in range(0, concat_step, 2):
            concat = tf.concat([conv1_concats_1[i], conv1_concats_1[i + 1]], axis=2)
            conv1_concats_2.append(concat)
        conv1_concats = tf.concat(conv1_concats_2, axis=0)
        return conv1_concats

    def build_graph(self):       
        #可视化
        self.writer = tf.summary.FileWriter("./logs",self.g)
       
        #图像输入层 
        self.x = tf.placeholder(tf.float32, shape=self.IMAGE_DIM, name='input') 
        # self.viwer_op.append(tf.summary.image("layer_0_input", self.x,max_outputs=10))


        #卷积层建立
            #                     #输入，卷积核个数，核大小，步进，激活函数，名称
            # h = tf.layers.conv2d(self.x, 24, 5, strides=2, activation=tf.nn.relu, name="conv1")       
            # # self.viwer_op.append(tf.summary.image("layer_1_Conv1", self.Viwer(24,h),max_outputs=10))

            # h = tf.layers.conv2d(h, 32, 5, strides=2, activation=tf.nn.relu, name="conv2")
            # # self.viwer_op.append(tf.summary.image("layer_2_Conv2", self.Viwer(32,h)))

            # h = tf.layers.conv2d(h, 64, 5, strides=2, activation=tf.nn.relu, name="conv3")
            # # self.viwer_op.append(tf.summary.image("layer_3_Conv3", self.Viwer(64,h)))
            # h = tf.layers.conv2d(h, 64, 3, strides=2, activation=tf.nn.relu, name="conv4")
            # # self.viwer_op.append(tf.summary.image("layer_4_Conv4", self.Viwer(64,h)))
            # h = tf.layers.conv2d(h, 64, 3, strides=1, activation=tf.nn.relu, name="conv5")
            # # self.viwer_op.append(tf.summary.image("layer_5_Conv5", self.Viwer(64,h)))

        #RNN
        n_hidden_units = 512*8
        # print("self.IMAGE_DIM:")
        # print(self.IMAGE_DIM)
        n_inputs = 256*3
        n_steps= 144
        # 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
        # batch_size_t = tf.placeholder(tf.int32)  # 注意类型必须为 tf.int32
        batch_size_t = 64
        weights = {
            # shape (28, 512)
            'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
            # shape (512, 15)
            'out': tf.Variable(tf.random_normal([n_hidden_units, 1024]))
        }
        biases = {
            # shape (512, )
            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
            # shape (15, )
            'out': tf.Variable(tf.constant(0.1, shape=[1024, ]))
        }
        
        X = tf.reshape(self.x, [-1, n_inputs])

        # X_in = W*X + b
        X_in = tf.matmul(X, weights['in']) + biases['in']
        X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

        # 使用 basic LSTM Cell.
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        init_state = lstm_cell.zero_state(batch_size_t, dtype=tf.float32) # 初始化全零 state

        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
        z = tf.matmul(outputs[-1], weights['out']) + biases['out']    #选取最后一个 output

        print(z.shape)
        # #降维度 如 [N*4*4] -> [N*16]
        # z = tf.layers.flatten(h)  
        #全连接层1，节点100,激活函数relu
        z = tf.layers.dense(z, 256, activation=tf.nn.relu, name="dense1")
        z = tf.nn.dropout(z, 0.9, name="dropout1")
        #全连接层2,节点50,激活函数relu
        z = tf.layers.dense(z, 64, activation=tf.nn.relu, name="dense2")
        z = tf.nn.dropout(z, 0.9, name="dropout2")
        #全连接层3_1,节点15,激活函数softmax 
        self.angle_out = tf.layers.dense(z, 15, activation=tf.nn.softmax, name="angle_out") # category probability 15
        #全连接层3_2,节点1,激活函数None                                                                
        self.throttle_out = tf.layers.dense(z, 1, activation=None, name="throttle_out")     #？？车速不可控制？？

        if self.is_training:
            #角度输入
            self.angle_target = tf.placeholder(tf.float32, shape=self.ANGLE_DIM, name='angle_target')
            #速度输入
            self.throttle_target = tf.placeholder(tf.float32, shape=self.THROTTLE_DIM, name='throttle_target')
            #loss
                #角度损失函数
            self.angle_loss = -tf.reduce_sum(self.angle_target * tf.log(tf.clip_by_value(self.angle_out, 1e-7, 1-1e-7)), axis=-1)
            self.angle_loss = tf.reduce_mean(self.angle_loss)
                #速度损失函数
            self.throttle_loss = tf.reduce_mean(tf.abs(self.throttle_target - self.throttle_out), axis=-1)      #???这里有问题？？？
            self.throttle_loss = tf.reduce_mean(self.throttle_loss)
                #总损失函数
            self.angle_weight = tf.Variable(0.5, trainable=False, name="angle_weight")
            self.throttle_weight = tf.Variable(0.5, trainable=False, name="throttle_weight")
            self.loss = self.angle_weight * self.angle_loss + self.throttle_weight * self.throttle_loss

            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.lr = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, 0.96, staircase=True)####### not sure if we shall add learning rate 
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def train(self, X_train, Y_train, X_val, Y_val, saved_model, epochs=100, exit_k=5, batch_size=32, new_model=True):
        if not new_model:
            self.load(saved_model)

        img_train = np.array(X_train[0])
        angle_train = np.array(Y_train[0])
        throttle_train = np.array(Y_train[1]).reshape(-1, 1)

        img_val = np.array(X_val[0])
        angle_val = np.array(Y_val[0])
        throttle_val = np.array(Y_val[1])

        total_train = len(img_train)
        # print(total_train)
        total_val = len(img_val)
        # print(total_val)

        img_val = img_val.reshape(-1, batch_size,cfg['CNN']['CNN_IMG_HEIGHT'],cfg['CNN']['CNN_IMG_WIDTH'], 3)
        # print(img_val.shape)
        angle_val = angle_val.reshape(-1, batch_size, 15)
        # print(img_val.shape)
        throttle_val = throttle_val.reshape(-1, batch_size, 1)

        train_steps = int(total_train // batch_size)
        val_steps = int(total_val // batch_size)
        # print(val_steps)
        # input("waiting")

        val_loss_min = 10000
        earlystop_num = 0
        earlystop_flag = 0

        for epoch in range(epochs):
            index = np.random.permutation(total_train)
            img_shuffle = img_train[index].reshape(-1, batch_size,cfg['CNN']['CNN_IMG_HEIGHT'],cfg['CNN']['CNN_IMG_WIDTH'],  3)
            angle_shuffle = angle_train[index].reshape(-1, batch_size, 15)
            throttle_shuffle = throttle_train[index].reshape(-1, batch_size, 1)

            for train_step in range(train_steps):
                img_batch = img_shuffle[train_step]
                angle_batch = angle_shuffle[train_step]
                throttle_batch = throttle_shuffle[train_step]
                feed = {self.x: img_batch, self.angle_target: angle_batch, self.throttle_target: throttle_batch}
                loss, angle_loss, throttle_loss, step, _ = self.sess.run([self.loss, self.angle_loss, self.throttle_loss, self.global_step, self.train_op], feed)
                #10个batch输出一次
                if (step+1) % 10 == 0:
                    output_log = "step: %d, loss: %.6f, angle_loss: %.6f, throttle_loss: %.6f" % ((step+1), loss, angle_loss, throttle_loss)
                    print(output_log)

            val_loss = val_angle_loss = val_throttle_loss = 0
            for val_step in range(val_steps):
                img_batch = img_val[val_step]
                angle_batch = angle_val[val_step]
                throttle_batch = throttle_val[val_step]
                val_feed = {self.x: img_batch, self.angle_target: angle_batch, self.throttle_target: throttle_batch}
                loss, angle_loss, throttle_loss = self.sess.run([self.loss, self.angle_loss, self.throttle_loss], val_feed)
                val_loss += loss
                val_throttle_loss += throttle_loss
                val_angle_loss += angle_loss

            val_loss /= val_steps
            val_angle_loss /= val_steps
            val_throttle_loss /= val_steps
            #每次整个数据集训练完后输出一次
            output_log = "epoch: %d, val_loss: %.6f, val_angle_loss: %.6f, val_throttle_loss: %.6f" % (epoch, val_loss, val_angle_loss, val_throttle_loss)
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
 
            #连续N次的损失大于之前的损失，则退出
            if earlystop_num >= exit_k:
                earlystop_flag = 1

            if earlystop_flag:
                print("Early Stop")
                break

    def run(self, img_arr):   
        img_arr = img_arr.reshape((1,) + img_arr.shape)
       
        feed = {self.x: img_arr}

        angle, throttle= self.sess.run([self.angle_out, self.throttle_out], feed)
        #pak=self.sess.run([self.angle_out, self.throttle_out]+self.viwer_op, feed)
        angle_unbinned = data.linear_unbin(angle[0])
        #print(pak[0][0])
        #angle_unbinned = data.linear_unbin(pak[0][0])           
        # i=2
        # while i<2+6:
        #     self.writer.add_summary(pak[i])
        #     i+=1
        #print(angle_unbinned)
        return angle_unbinned, throttle[0][0]

