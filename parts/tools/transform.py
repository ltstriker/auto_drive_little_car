from PIL import Image
import random
import numpy as np
from core.config import load_config
cfg=load_config()
# 进行图像的预处理

def transform(img_batch):
    img_batch = img_batch.reshape(-1, cfg['CNN']['CNN_IMG_HEIGHT'], cfg['CNN']['CNN_IMG_WIDTH'],  3)
    for i in range(len(img_batch)):
        img = Image.fromarray(img_batch[i])
        if(random.random()<1):
            ranH = np.random.randn()*10+10
            if(ranH>20):
                ranH = 20
            if(ranH<0):
                ranH = 0
            crop =  (0, 0+ranH, 256, 124+ranH)
            trans1 =img.transform((256,144), Image.EXTENT, crop)
            # trans1.show()
            
        if(random.random()<1):
            rand = random.random()
            rotate = np.random.randn()*30
            if(rotate>30):
                rotate = 30
            if(rotate<-30):
                rotate = -30
            if(rotate>=0):
                trans3 = trans1.transform((256,144),  Image.QUAD, (0,rotate,rotate,144,256,144-rotate,256-rotate,0))
            if(rotate<0):
                rotate = -rotate
                trans3 = trans1.transform((256,144),  Image.QUAD, (rotate,0,0,144-rotate,256-rotate,144,256,rotate))
            # trans3.show()
        imgnumpy = np.array(trans3)
        img_batch[i] = imgnumpy
    img_batch = img_batch.reshape(cfg['TRAINING']['BATCH_SIZE'], cfg['TRAINING']['SEQUENCE_LENGTH'], cfg['CNN']['CNN_IMG_HEIGHT'],cfg['CNN']['CNN_IMG_WIDTH'],  3)
    return img_batch