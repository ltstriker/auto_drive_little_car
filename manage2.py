#!/usr/bin/env python3
"""
manage.py
The main module that control the
"""
import os,sys
# AuraPackPath=os.path.abspath(os.curdir)+"/../"
# sys.path.append(AuraPackPath)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse

#from docopt import docopt

from core.config import load_config
from core.vehicle import Vehicle

from parts.nets.tfRNN import CNN,reset_graph
from parts.sensor.camera import Webcam
from parts.controller.controller import JoystickController
from parts.controller.actuator import PCA9685, PWMSteering, PWMThrottle
from parts.controller.transform import Lambda
from parts.tools.datastore import TubHandler, TubGroup, Tub
from parts.tools import data


class BaseCommand():
    pass

class Drive(BaseCommand):
    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='drive', usage='%(prog)s [options]')
        parser.add_argument('--model', help='The model used for auto driving')
        parsed_args = parser.parse_args(args)
        return parsed_args    

    def run(self, args):
        cfg = load_config()
        if args:
            args = self.parse_args(args)
            self.drive(cfg=cfg,model_path=args.model,)
        else:
            self.drive(cfg)
    def drive(self,cfg, model_path=None):

        #Initialize car
        V = Vehicle()

        cam = Webcam(resolution=(cfg['CAMERA']['CAMERA_RESOLUTION']['HEIGHT'],cfg['CAMERA']['CAMERA_RESOLUTION']['WIDTH']),framerate=cfg['CAMERA']['CAMERA_FRAMERATE'])
        V.add(cam, outputs=['cam/image_array'], threaded=True)
    
  
        ctr = JoystickController(max_throttle=cfg['JOYSTICK']['JOYSTICK_MAX_THROTTLE'],
                                    steering_scale=cfg['JOYSTICK']['JOYSTICK_STEERING_SCALE'],
                                    auto_record_on_throttle=cfg['JOYSTICK']['AUTO_RECORD_ON_THROTTLE'])                         

        V.add(ctr, 
            inputs=['cam/image_array'],
            outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
            threaded=True)
        
        
        
        
        #See if we should even run the pilot module. 
        #This is only needed because the part run_condition only accepts boolean
        def pilot_condition(mode):
            if mode == 'user':
                return False
            else:
                return True
        pilot_condition_part = Lambda(pilot_condition)
        V.add(pilot_condition_part, inputs=['user/mode'], outputs=['run_pilot'])
        
        #Run the pilot if the mode is not user.

        reset_graph()
        CNN_model = CNN(is_training=False)
        if model_path:
            CNN_model.load(model_path)
    
        V.add(CNN_model, inputs=['cam/image_array'],
            outputs=['pilot/angle', 'pilot/throttle'],
            run_condition='run_pilot')
  
        #Choose what inputs should change the car.
        def drive_mode(mode, 
                    user_angle, user_throttle,
                    pilot_angle, pilot_throttle):
            if mode == 'user': 
                return user_angle, user_throttle
            
            elif mode == 'local_angle':
                return pilot_angle, user_throttle
            
            else: 
                return pilot_angle, pilot_throttle
            
        drive_mode_part = Lambda(drive_mode)
        V.add(drive_mode_part, 
            inputs=['user/mode', 'user/angle', 'user/throttle',
                    'pilot/angle', 'pilot/throttle'], 
            outputs=['angle', 'throttle'])

        steering_controller=PCA9685(cfg['STEERING']['STEERING_CHANNEL'])
        steering =PWMSteering(controller=steering_controller,left_pulse=cfg['STEERING']['STEERING_LEFT_PWM'],
		      right_pulse=cfg['STEERING']['STEERING_RIGHT_PWM'])

        throttle_controller=PCA9685(cfg['THROTTLE']['THROTTLE_CHANNEL'])
        throttle =PWMThrottle(controller=throttle_controller,max_pulse=cfg['THROTTLE']['THROTTLE_FORWARD_PWM'],
                              zero_pulse=cfg['THROTTLE']['THROTTLE_STOPPED_PWM'],min_pulse=cfg['THROTTLE']['THROTTLE_REVERSE_PWM'])
        V.add(steering,inputs=['angle'])
        V.add(throttle,inputs=['throttle'])


        #add tub to save data
        inputs=['cam/image_array', 'user/angle', 'user/throttle', 'user/mode']
        types=['image_array', 'float', 'float',  'str']
        

        th = TubHandler(path="./data")
        tub = th.new_tub_writer(inputs=inputs, types=types)
        V.add(tub, inputs=inputs, run_condition='recording')
    
        #run the vehicle
        V.start(rate_hz=cfg['VEHICLE']['DRIVE_LOOP_HZ'], 
                max_loop_count=cfg['VEHICLE']['MAX_LOOPS'])

class Train(BaseCommand):
    import tensorflow as tf
    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='train', usage='%(prog)s [options]')
        parser.add_argument('--tub', help='data for auto drive training')
        parser.add_argument('--model', help='model path for saving new models')
        parser.add_argument('--base_model', help='base model path')
        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        cfg =load_config()
        args = self.parse_args(args)
        self.train(cfg, tub_names=args.tub, new_model_path=args.model,base_model_path=args.base_model)

    def train(self, cfg, tub_names, new_model_path, base_model_path=None):
        """
        use the specified data in tub_names to train an artifical neural network
        saves the output trained model as model_name
        """
        X_keys = ['cam/image_array']
        y_keys = ['user/angle', 'user/throttle']
        def train_record_transform(record):
            """ convert categorical steering to linear and apply image augmentations """
            record['user/angle'] = data.linear_bin(record['user/angle'])
            # TODO add augmentation that doesn't use opencv
            return record

        def val_record_transform(record):
            """ convert categorical steering to linear """
            record['user/angle'] = data.linear_bin(record['user/angle'])
            return record
   
        if not tub_names:
            tub_names = os.path.join("./data", '*')
        tubgroup = TubGroup(tub_names)
        X_train, Y_train, X_val, Y_val = tubgroup.get_train_val_gen(X_keys, y_keys,
                                                                    record_transform=train_record_transform,
                                                                    batch_size=cfg['TRAINING']['BATCH_SIZE'],
                                                                    train_frac=cfg['TRAINING']['TRAIN_TEST_SPLIT'])
        print('tub_names', tub_names)

        total_records = len(tubgroup.df)
        total_train = int(total_records * cfg['TRAINING']['TRAIN_TEST_SPLIT'])
        total_val = total_records - total_train
        print('train: %d, validation: %d' % (total_train, total_val))
        steps_per_epoch = total_train // cfg['TRAINING']['BATCH_SIZE']
        print('steps_per_epoch', steps_per_epoch)

        new_model_path = os.path.expanduser(new_model_path)
        reset_graph()
        CNN_model = CNN(is_training=True, learning_rate=0.001)
        if base_model_path is not None:
            base_model_path = os.path.expanduser(base_model_path)
            #tfcategorical.load_tensorflow(base_model_path)
        CNN_model.train(X_train, Y_train, X_val, Y_val, saved_model=new_model_path,epochs=30, 
                                batch_size=cfg['TRAINING']['BATCH_SIZE'], new_model=True)
        CNN_model.close_sess()

class CalibrateCar(BaseCommand):

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='calibrate', usage='%(prog)s [options]')
        parser.add_argument('--channel', help='The channel youd like to calibrate [0-15]')
        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        from parts.controller.actuator import PCA9685

        args = self.parse_args(args)
        channel = int(args.channel)
        c = PCA9685(channel)

        for i in range(10):
            pmw = int(input('Enter a PWM setting to test(0-1500)'))
            c.run(pmw)

class MakeMovie(BaseCommand):

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='makemovie')
        parser.add_argument('--tub', help='The tub to make movie from')
        parser.add_argument('--out', default='tub_movie.mp4', help='The movie filename to create. default: tub_movie.mp4')
        parsed_args = parser.parse_args(args)
        return parsed_args, parser

    def run(self, args):
        """
        Load the images from a tub and create a movie from them.
        Movie
        """
        import moviepy.editor as mpy

        args, parser = self.parse_args(args)

        if args.tub is None:
            parser.print_help()
            return

        cfg = load_config()

        self.tub = Tub(args.tub)
        self.num_rec = self.tub.get_num_records()
        self.iRec = 0

        print('making movie', args.out, 'from', self.num_rec, 'images')
        clip = mpy.VideoClip(self.make_frame, duration=(self.num_rec//cfg['VEHICLE']['DRIVE_LOOP_HZ']) - 1)
        clip.write_videofile(args.out,fps=cfg['VEHICLE']['DRIVE_LOOP_HZ'])

        print('done')

    def make_frame(self, t):
        """
        Callback to return an image from from our tub records.
        This is called from the VideoClip as it references a time.
        We don't use t to reference the frame, but instead increment
        a frame counter. This assumes sequential access.
        """
        self.iRec = self.iRec + 1

        if self.iRec >= self.num_rec - 1:
            return None

        rec = self.tub.get_record(self.iRec)
        image = rec['cam/image_array']

        return image # returns a 8-bit RGB array

def execute_from_command_line():
    """
    This is the function linked to the "senserover" terminal command.
    """
    commands = {
            'drive': Drive,
            'train': Train,
            'calibrate': CalibrateCar,
            'makemovie': MakeMovie,
                }
    args = sys.argv[:]
    command_text = args[1]

    if command_text in commands.keys():
        command = commands[command_text]
        c = command()
        c.run(args[2:])

if __name__ == '__main__':
    execute_from_command_line()
