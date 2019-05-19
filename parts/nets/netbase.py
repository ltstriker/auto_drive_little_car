#'''

# netbase.py
#
# netbase.py builds up the fundamental parts for the network.
# It is inherited by the net.py which contains specific network that works for the training.

#'''

import numpy as np
import json
import pickle
import tensorflow as tf
from PIL import Image
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class Netbase(object):
    def __init__(self, is_training=True, net_type=None, *args, **kwargs):
        self.is_training = is_training
        self.net_type = net_type

    def build_graph(self):
        # build graph in the net
        pass

    def reset_graph(self):
        if 'sess' in globals() and self.sess:
            self.sess.close()
        tf.reset_default_graph()

    def close_sess(self):
        # Close tensorflow session
        self.sess.close()

    def get_model_params(self):
        # get trainable parameters
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                param_name = var.name
                p = self.sess.run(var)
                model_names.append(param_name)
                # p*10000 means to enlarge the parameter so that the precision would not be lost during training
                params = np.round(p * 10000).astype(np.int).tolist()
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names

    def set_model_params(self, params):
        # set trainable parameters
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                pshape = self.sess.run(var).shape
                p = np.array(params[idx])
                assert pshape == p.shape, "inconsistent shape"
                assign_op = var.assign(p.astype(np.float) / 10000.)
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

    def load_pickle(self, picklefile):
        with open(picklefile, 'rb') as f:
            params = pickle.load(f, encoding='latin1')
        with self.g.as_default():
            train_vars = tf.trainable_variables()
            for var in train_vars:
                var_shape = self.sess.run(var).shape
                var_name = var.name.split(':')[0]
                if "kernel" in var_name:
                    var_name = var_name.split('/')[0] + ".w"
                if "bias" in var_name:
                    var_name = var_name.split('/')[0] + ".b"
                param = np.array(params[var_name])
                assert var_shape == param.shape, "inconsistent shape"
                assign_op = var.assign(param.astype(np.float32))
                self.sess.run(assign_op)

    def load_parrots(self, model_path):
        try:
            self.load_pickle(model_path)
        except:
            print("Failed to load Parrots parameters")

    def load_tensorflow(self, model_path):
        try:
            self.load_json(model_path)
        except:
            print("Error: cannot load model file!")

###############VAE-RNN###############

    def load_image(self, name):
        image = Image.open(name)
        image = image.convert('RGB')
        image = np.array(image)
        return image

    def load_action_json(self, filename):
        with open(filename, 'r') as f:
            params = json.load(f)
        throttle = params["user/throttle"]
        angle = params["user/angle"]
        return [throttle, angle]

    def load_image_dataset(self, dir):
        print("Start Loading...")
        f = []
        for (dirpath, dirname, filename) in os.walk(dir):
            f.extend(filename)
        images = []
        for i, name in enumerate(f):
            if 'jpg' in name:
                im = self.load_image(os.path.join(dir, name))
                images.append(im)
                if (i + 1) % 100 == 0:
                    print("Already load {} images...".format(i + 1))
        print("Total Images:", len(images))
        images = np.array(images)
        return images

    def load_series_dataset(self, dir):
        print("Start Loading...")
        f = []
        for (dirpath, dirname, filename) in os.walk(dir):
            f.extend(filename)
        images = []
        labels = []
        for i, name in enumerate(f):
            if 'jpg' in name:
                im = self.load_image(os.path.join(dir, name))
                images.append(im)
                file_index = name.split('_')[0]
                json_file = "record_" + file_index + ".json"
                l = self.load_action_json(os.path.join(dir, json_file))
                labels.append(l)
                if (i + 1) % 100 == 0:
                    print("Already load {} data...".format(i + 1))
        print("Total Data:", len(images))
        images = np.array(images)
        labels = np.array(labels)
        return images, labels

    def load_rnn_dataset(self, file):
        raw_data = np.load(file)
        input = raw_data['input']
        label = raw_data['label']
        return input, label

    def run(self, *args, **kwargs):
        # Run the net for pilot
        pass

    def shutdown(self):
        # Shut down the net
        pass