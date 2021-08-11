#!/usr/bin/env python3
from playsound import playsound
import os

def beep():
    try:
        playsound(os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/../../../assets/audio/beep.ogg"))
    except:
        pass

def boop():
    try:
        playsound(os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/../../../assets/audio/boop.ogg"))
    except:
        pass
