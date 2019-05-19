__version__ = '1.0.0'

print('using SenseRover v{} ...'.format(__version__))

import sys

if sys.version_info.major < 3:
    msg = 'Aura Requires Python 3.4 or greater. You are using {}'.format(sys.version)
    raise ValueError(msg)


from .core import memory
from .core import vehicle
from .core import config
from .core.config import load_config
from .parts import *