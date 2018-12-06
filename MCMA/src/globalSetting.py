"""
global Variables
"""
import sys
import multiprocessing as mp
import os
import time
import math
import random

import glob
import logging
import argparse
import json
import operator
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Time in this format [weekday, month, day, h:m:s, year], for log
localtime = time.asctime(time.localtime(time.time()))
month, day = localtime.split()[1:3]

# default paths
configPath = "../configs/sample.json"
workDir = "../runs/"+month+day+'/'
dataDir = "../data/"
logPath = workDir + "log.txt"

if (os.path.exists(workDir)==False):
    os.mkdir(workDir)

# global cuda flag
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# easy tensor
mytensor = lambda arr, varType: torch.tensor(arr, device=device, dtype=varType)

# For training, other variables

benchName = 'bessel_Jnu'
c = {}
trainSrc = []
trainTgt = []
testSrc = []
testTgt = []
netA = []
netC = []


