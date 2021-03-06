import os,sys
import argparse
import json

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from core.config import load_config
from parts.nets.tfRNN import CNN,reset_graph
from parts.tools.sViewr import sViewr
from PIL import Image

class BaseCommand():
    pass

class View(BaseCommand):
    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='view', usage='%(prog)s [options]')
        parser.add_argument('--model', help='The model used for auto driving')
        parser.add_argument('--tub', help='data for auto drive training')
        parsed_args = parser.parse_args(args)
        return parsed_args    

    def run(self, args):
        cfg = load_config()
        if args:
            args = self.parse_args(args)
            self.view(cfg=cfg,model_path=args.model,tub=args.tub)
        else:
            self.view(cfg)
    def view(self,cfg, model_path=None, tub=None):

        reset_graph()
        CNN_model = CNN(is_training=False)
        if model_path:
            print(model_path)
            CNN_model.load(model_path)
        sviewer=sViewr()
        tubs = os.listdir(tub)
        tubs = list(filter(lambda x:x.endswith('jpg'),tubs))
        tubs.sort(key=lambda x:int(x[:-21]))
        index = 8304
        filename = 'data/data'
        for etub in tubs:
            path = tub+'/'+etub
            print(tub,etub,path)
            img_PIL = Image.open(path)
            img_PIL.save(filename + '/' + str(index) + '_cam-image_array_.jpg')
            # angle, throttle = CNN_model.run(img_PIL_Tensor)
            # sviewer.run(path, angle, throttle)
            string=""
            for i in path.split('/')[:-1]:
                string+=i+"/"
                print(string)
            snum=path.split('/')[-1].split('_')[0]

            f=open(string+"record_"+snum+".json")
            jsonfile = json.load(f)
            jsonpath = filename + '/' + 'record_' + str(index) + 'json'
            json.dump(jsonfile,jsonpath,ensure_ascii=False)
            index = index + 1
def execute_from_command_line():

    commands = {
            'view': View,
                }
    args = sys.argv[:]
    command_text = args[1]

    if command_text in commands.keys():
        command = commands[command_text]
        c = command()
        c.run(args[2:])

if __name__ == '__main__':
    execute_from_command_line()