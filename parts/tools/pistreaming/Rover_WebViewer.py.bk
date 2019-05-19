#!/usr/bin/env python

import sys
import io
import os
import shutil
from subprocess import Popen, PIPE
from string import Template
from struct import Struct
from threading import Thread
from time import sleep, time
#import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from wsgiref.simple_server import make_server

import picamera
from ws4py.websocket import WebSocket
from ws4py.server.wsgirefserver import (
    WSGIServer,
    WebSocketWSGIHandler,
    WebSocketWSGIRequestHandler,
)
from ws4py.server.wsgiutils import WebSocketWSGIApplication

import json

###########################################
# CONFIGURATION
WIDTH = 320
HEIGHT = 240
#FRAMERATE = 24
HTTP_PORT = 8082
WS_CONTROLLER_PORT=8083
WS_PORT = 8084

COLOR = u'#444'
BGCOLOR = u'#333'
JSMPEG_MAGIC = b'jsmp'
JSMPEG_HEADER = Struct('>4sHH')
VFLIP = False
HFLIP = False

###########################################
#for test
i=0
def changedata():
    while True:
        global i
        i=i+1
        sleep(0.04)
#

class StreamingHttpHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
            return
        elif self.path == '/jsmpg.js':
            content_type = 'application/javascript'
            content = self.server.jsmpg_content
        elif self.path == '/index.html':
            content_type = 'text/html; charset=utf-8'
            tpl = Template(self.server.index_template)
            content = tpl.safe_substitute(dict(
                WS_PORT=WS_PORT,WS_CONTROLLER_PORT=WS_CONTROLLER_PORT, WIDTH=WIDTH, HEIGHT=HEIGHT, COLOR=COLOR,
                BGCOLOR=BGCOLOR))
        else:
            self.send_error(404, 'File not found')
            return
        content = content.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', len(content))
        self.send_header('Last-Modified', self.date_time_string(time()))
        self.end_headers()
        if self.command == 'GET':
            self.wfile.write(content)


class StreamingHttpServer(HTTPServer):
    def __init__(self):
        super(StreamingHttpServer, self).__init__(
                ('', HTTP_PORT), StreamingHttpHandler)
        this_dir = os.path.dirname(os.path.realpath(__file__))
        with io.open(this_dir+'/index.html', 'r') as f:
            self.index_template = f.read()
        with io.open(this_dir+'/jsmpg.js', 'r') as f:
            self.jsmpg_content = f.read()


class StreamingWebSocket(WebSocket):
    def opened(self):
        self.send(JSMPEG_HEADER.pack(JSMPEG_MAGIC, WIDTH, HEIGHT), binary=True)

class ControllerStreamingWebSocket(WebSocket):
    def opened(self):
        print("ws_cd_opened")
        self.send("ControllerSocket opened successful!,from server")

    def closed(self,*arg):
        print("ws_cd_closed")

class ControllerBroadcast():
    def __init__(self,controller_websocket_server):
        self.controller_websocket_server=controller_websocket_server
        self.data=" "
    def updata(self):
        while True:
            global i
            if(len(self.data)>0):
                self.controller_websocket_server.manager.broadcast(self.data)
            sleep(0.05)


class BroadcastOutput(object):
    def __init__(self, camera):
        print('Spawning background conversion process')
        self.converter = Popen([
            'ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'yuv420p',
            '-s', '%dx%d' % camera.resolution,
            '-r', str(float(camera.framerate)),
            '-i', '-',
            '-f', 'mpeg1video',
            '-b', '800k',
            '-r', str(float(camera.framerate)),
            '-'],
            stdin=PIPE, stdout=PIPE, stderr=io.open(os.devnull, 'wb'),
            shell=False, close_fds=True)

    def write(self,b):
        self.converter.stdin.write(b)

    def flush(self):
        print('Waiting for background conversion process to exit')
        self.converter.stdin.close()
        self.converter.wait()


class BroadcastThread(Thread):
    def __init__(self, converter, websocket_server):
        super(BroadcastThread, self).__init__()
        self.converter = converter
        self.websocket_server = websocket_server

    def run(self):
        try:
            while True:
                buf = self.converter.stdout.read1(32768)
                if buf:
                    self.websocket_server.manager.broadcast(buf, binary=True)
                elif self.converter.poll() is not None:
                    break
        finally:
            self.converter.stdout.close()


class Rover_Viewer():
    def __init__(self,pi_camera,resolution=(320,240)):
        self.camera=pi_camera

        print('Viewer:Initializing websockets server server for img on port %d' % WS_PORT)
        WebSocketWSGIHandler.http_version = '1.1'
        self.websocket_server = make_server(
            '', WS_PORT,
            server_class=WSGIServer,
            handler_class=WebSocketWSGIRequestHandler,
            app=WebSocketWSGIApplication(handler_cls=StreamingWebSocket))
        self.websocket_server.initialize_websockets_manager()
        self.websocket_thread = Thread(target=self.websocket_server.serve_forever)

        print('Viewer:Initializing websockets server for control data on port %d' % WS_CONTROLLER_PORT)
        self.controller_websocket_server = make_server(
            '',WS_CONTROLLER_PORT,
            server_class=WSGIServer,
            handler_class=WebSocketWSGIRequestHandler,
            app=WebSocketWSGIApplication(handler_cls=ControllerStreamingWebSocket))
        self.controller_websocket_server.initialize_websockets_manager()
        self.controller_websocket_thread = Thread(target=self.controller_websocket_server.serve_forever)
        self.controller_broadcast=ControllerBroadcast(self.controller_websocket_server)
        self.controller_broadcast_thread=Thread(target=self.controller_broadcast.updata)

        print('Viewer:Initializing HTTP server on port %d' % HTTP_PORT)
        self.http_server = StreamingHttpServer()
        self.http_thread = Thread(target=self.http_server.serve_forever)

        print('Viewer:Initializing broadcast thread')
        self.output = BroadcastOutput(self.camera)
        self.broadcast_thread = BroadcastThread(self.output.converter,self.websocket_server)
    
    def update(self):
        self.camera.start_recording(self.output, 'yuv',resize=(WIDTH,HEIGHT))

        print('Viewer:Starting HTTP server thread')
        self.http_thread.start()

        print('Viewer:Viewer:Starting websockets for img thread')
        self.websocket_thread.start()

        print('Viewer:Starting websockets for control data thread')
        self.controller_websocket_thread.start()
        self.controller_broadcast_thread.start()

        print('Viewer:Starting broadcast thread')
        self.broadcast_thread.start()

#for debug
        # print('Viewer:test thread start')
        # t=Thread(target=changedata)
        # t.start()
#for debug

        while True:
            self.camera.wait_recording(0.5)
    
    def shutdown(self):
        print('Viewer:Stopping recording')
        self.camera.stop_recording()
        print('Viewer:Waiting for broadcast thread to finish')
        self.broadcast_thread.join()
        print('Viewer:Shutting down HTTP server')
        self.http_server.shutdown()
        print('Viewer:Shutting down websockets server')
        self.websocket_server.shutdown()
        print('Viewer:Shutting down websockets server')
        self.controller_websocket_server.shutdown()
        print('Viewer:Waiting for HTTP server thread to finish')
        self.http_thread.join()
        print('Viewer:Waiting for websockets thread to finish')
        self.websocket_thread.join()

    def run(self):
        return None
    
    def run_threaded(self,angle=1.00,throttle=1.00):
        if angle==None:return None
        angle=float('%.2f'% angle)
        throttle=float('%.2f'% throttle)
        self.controller_broadcast.data=json.dumps([{"angle":str(angle),"throttle":str(throttle)}])
#        self.controller_broadcast.data=str(angle)
        return None
