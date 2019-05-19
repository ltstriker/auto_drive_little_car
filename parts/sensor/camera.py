import os
import time
import numpy as np
from PIL import Image
import glob
import pygame
class BaseCamera:

    def run_threaded(self):
            return self.frame

class PiCamera(BaseCamera):
    def __init__(self, save_resolution=(120, 160), framerate=20):
        from picamera.array import PiRGBArray
        from picamera import PiCamera
     
        # initialize the camera and stream
        resolution = (320,240)   #inintal size of camera 

        if save_resolution[1]>320 or save_resolution[0]>240:
            self.save_resolution=(240,320)
            print("target img size should not over 320*240")
        else:
            self.save_resolution=save_resolution #saved size

        self.camera = PiCamera() #PiCamera gets resolution (height, width)
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="rgb", use_video_port=True)

        
        self.ROI_X_0=(320-self.save_resolution[1])/2
        self.ROI_Y_0=(240-self.save_resolution[0])/2
        self.ROI_X_1=320-(320-self.save_resolution[1])/2
        self.ROI_Y_1=240-(240-self.save_resolution[0])/2
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.on = True

        print('PiCamera loaded.. .warming camera')
        time.sleep(2)


    def run(self):
        f = next(self.stream)
        frame = f.array
        self.rawCapture.truncate(0)
        return frame

    def update(self):
       
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            frame = f.array
            f_rs=Image.fromarray(np.uint8(frame))
            #f_rs=f_rs.resize((160,120),Image.ANTIALIAS)
            f_rs=f_rs.crop((self.ROI_X_0,self.ROI_Y_0,self.ROI_X_1,self.ROI_Y_1))
            self.frame=np.asarray(f_rs)
            self.rawCapture.truncate(0)
            
            # if the thread indicator variable is set, stop the thread
            if not self.on:
                break

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        print('stoping PiCamera')
        time.sleep(.5)
        self.stream.close()
        self.rawCapture.close()
        self.camera.close()

class Webcam(BaseCamera):
    def __init__(self, resolution = (160, 120), framerate = 20):
        import pygame
        import pygame.camera

        super().__init__()

        pygame.init()
        pygame.camera.init()
        l = pygame.camera.list_cameras()

        #设置焦距为无限远
        os.system('v4l2-ctl -d 0 -c focus_auto=0')

        #这里可以改进一下，在线程中初始化，然后再启动，避免没有摄像头带来的错误
        if(l!=None):
            print("using camera:"+str(l[0]))
            self.cam = pygame.camera.Camera(l[0], (640,360), "RGB")
            self.cam.start()
        else:
            print("can not find camera")
            
        self.resolution = resolution
        self.framerate = framerate

        #设置显示窗口
        self.display=pygame.display.set_mode((640,360), 0)


        # initialize variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.on = True

        print('WebcamVideoStream loaded.. .warming camera')

        time.sleep(2)

    def update(self):
        from datetime import datetime, timedelta
        import pygame.image

        flag=1

        while self.on:
            start = datetime.now()

            if self.cam.query_image():
                snapshot = self.cam.get_image()
                #确定第一张图的分辨率
                if flag==1:
                    #pygame.image.save(snapshot, "img.jpg")
                    print("size:",self.resolution)
                    print("real_size:",self.cam.get_size())
                    flag=0
                
                snapshot1 = pygame.transform.scale(snapshot, self.resolution)

                self.display.blit(snapshot, (0,0))
                pygame.display.flip()

                self.frame = pygame.surfarray.pixels3d(pygame.transform.rotate(pygame.transform.flip(snapshot1, True, False), 90))
            stop = datetime.now()
            s = 1 / self.framerate - (stop - start).total_seconds()
            if s > 0:
                time.sleep(s)

        self.cam.stop()

    def run_threaded(self):
        return self.frame

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        print('stoping Webcam')
        time.sleep(.5)

class ImageListCamera(BaseCamera):
    '''
    Use the images from a tub as a fake camera output
    '''
    def __init__(self, path_mask='~/d2/data/**/*.jpg',StartView=False):
        self.image_filenames = glob.glob(os.path.expanduser(path_mask), recursive=True)
    
        def get_image_index(fnm):
            sl = os.path.basename(fnm).split('_')
            return int(sl[0])

        '''
        I feel like sorting by modified time is almost always
        what you want. but if you tared and moved your data around,
        sometimes it doesn't preserve a nice modified time.
        so, sorting by image index works better, but only with one path.
        '''
        self.image_filenames.sort(key=get_image_index)
        #self.image_filenames.sort(key=os.path.getmtime)
        self.num_images = len(self.image_filenames)
        print('%d images loaded.' % self.num_images)
        print( self.image_filenames[:10])
        self.i_frame = 0
        self.frame = None
        self.update()

        #for viwer
        self.startView=StartView
        if(self.startView):
            pygame.init()
            self.display=pygame.display.set_mode((256,144), 0)
        else:
            self.display=None

    def update(self):
        pass

    def run_threaded(self):        
        if self.num_images > 0:
            self.i_frame = (self.i_frame + 1) % self.num_images
            self.frame = Image.open(self.image_filenames[self.i_frame]) 
            
            #for viwer
            if(self.startView):
                img = pygame.image.load(self.image_filenames[self.i_frame])
                self.display.blit(img, (0,0))
                pygame.display.flip()
        return np.asarray(self.frame),self.image_filenames[self.i_frame]

    def shutdown(self):
        pass
