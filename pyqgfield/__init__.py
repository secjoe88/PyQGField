from pyqgfield.items.GLQuiverItem import GLQuiverItem
from pyqgfield.items.GLArrowItem import GLArrowItem
from pyqgfield.items.GLParticleItem import GLParticleItem
from pyqtgraph import *
from numpy import *
import logging


__all__=['items']

logging.basicConfig(
    filename='c:\\users\\jojo\\desktop\\log', 
    format='[%(asctime)s] %(levelname)s\t[%(name)s.%(funcName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
    
logging.info(
'************************New PyQGField Session Started************************'
)