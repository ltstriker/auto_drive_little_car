import os
import time
import pygame
import math
import json

from core.config import load_config
cfg=load_config()

class sViewr():
    def __init__(self):
        pygame.init()
        # self.display=pygame.display.set_mode((256,144), 0)

    def update(self):
        pass
    
    def run(self,path,angle):
        if angle!=None:
            img = pygame.image.load(path)
            
            #red line for virtual angle
            #print(angle)
            
            x=128+20*math.sin(angle/1*math.pi/4)
            y=100+20-20*math.cos(angle/1*math.pi/4)
            pygame.draw.line(img, (255,0,0), (128,144), (x,y), 3)


            string=""
            for i in path.split('/')[:-1]:
                string+=i+"/"
            snum=path.split('/')[-1].split('_')[0]

            f=open(string+"record_"+snum+".json")
            rangle=json.loads(f.read())["user/angle"]
            f.close()
            
            #green line for real angle 
            rx=128+20*math.sin(rangle/1*math.pi/4)
            ry=100+20-20*math.cos(rangle/1*math.pi/4)
            pygame.draw.line(img, (0,255,0), (128,144), (rx,ry), 3)
            
            pygame.image.save(img,"outputdata/"+snum+".jpeg")
            # self.display.blit(img, (0,0))
            # pygame.display.flip()
    
    def shutdown(self):
        pass