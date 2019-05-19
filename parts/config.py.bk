"""
Configuration
"""
from easydict import EasyDict as edict

cfg = edict()

# VAE options
cfg.VAE = edict()

cfg.VAE.IMG_HEIGHT = 144

cfg.VAE.IMG_WIDTH = 256

cfg.VAE.EPOCHS = 20

cfg.VAE.LEARNING_RATE = 0.0001

cfg.VAE.TF_ALLOW_GROWTH = True

cfg.VAE.BATCH_SIZE = 32

cfg.VAE.VAL_BATCH_SIZE = 100

cfg.VAE.LR_DECAY_STEPS = 1000

cfg.VAE.LR_DECAY_RATE = 0.9

cfg.VAE.Z_DIM = 128

cfg.VAE.IMG_CROP_SIZE = [120, 120]

cfg.VAE.MODEL_SAVE_DIR = "model"

cfg.VAE.TBOARD_SAVE_DIR = "tboard"

cfg.VAE.SERIES_SAVE_DIR = "series"


# RNN options
cfg.RNN = edict()

cfg.RNN.INWIDTH = 34

cfg.RNN.OUTWIDTH = 2

cfg.RNN.MAX_SEQ = 49

cfg.RNN.VAL_MAX_SEQ = 1

cfg.RNN.LAYER_NORM = False

cfg.RNN.EPOCHS = 200

cfg.RNN.LEARNING_RATE = 0.0001

cfg.RNN.TF_ALLOW_GROWTH = True

cfg.RNN.BATCH_SIZE = 10

cfg.RNN.VAL_BATCH_SIZE = 1

cfg.RNN.LR_DECAY_STEPS = 5000

cfg.RNN.LR_DECAY_RATE = 0.96

cfg.RNN.GRAD_CLIP = 10.0

cfg.RNN.MODEL_SAVE_DIR = "model"

cfg.RNN.TBOARD_SAVE_DIR = "tboard"

# CNN options
cfg.CNN = edict()

cfg.CNN.ANGLE_WEIGHT = 0.5

cfg.CNN.THROTTLE_WEIGHT = 1 - cfg.CNN.ANGLE_WEIGHT

cfg.CNN.LEARNING_RATE = 0.001

cfg.CNN.BATCH_SIZE = 32

cfg.CNN.TRAIN_VAL_SPLIT = 0.9

#only for mange_cnn.py
cfg.CNN.CNN_IMG_WIDTH=120
cfg.CNN.CNN_IMG_HEIGHT=160
