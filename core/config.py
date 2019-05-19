#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:27:44 2017

@author: wroscoe
"""
import os
import types
import yaml
def load_config(config_path=None):
    
    if config_path is None:
        import __main__ as main
        main_path = os.path.dirname(os.path.realpath(main.__file__))
        config_path = os.path.join(main_path, 'config.yml')
    print('loading config file: {}'.format(config_path))
    try:
        with open("config.yml") as setting_file:
            cfg=yaml.load(setting_file)
            print('config loaded')
    except IOError as e:
        pass
    return cfg
