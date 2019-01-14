import numpy as np
import os
import cv2
import json
import time
import base64

import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
# import matplotlib; matplotlib.use('Agg')
#import matplotlib
#matplotlib.use("Pdf")
from matplotlib import pyplot as plt
from PIL import Image

import glob
import xml.etree.ElementTree as ET
from PIL import Image
import paramiko

def sshCommand(ssh, command):
    
    stdin, stdout, stderr = ssh.exec_command(command)
    # print('Exec time: %s' % (time.time() - start))
    # print(stdin)
    # print(stderr.read())
    out = stdout.read()
    # print(stderr.read())
    # sshClient.close()
    if out:
        return out
    else:
        return False

def sshClient(hostname, port, username, password):
    sshClient = paramiko.SSHClient() # create SSHClient instance

    sshClient.set_missing_host_key_policy(paramiko.AutoAddPolicy())    # AutoAddPolicy automatically adding the hostname and new host key
    sshClient.load_system_host_keys()
    sshClient.connect(hostname, port, username, password)
    
    return sshClient

def decode_array_from_bytes(string):
    np_arr = np.fromstring(string, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def base64_decode_image(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    print(type(a))
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")
 
    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)
    # a = cv2.imdecode(a, cv2.IMREAD_COLOR)
 
    # return the decoded image
    return a


try:
    ssh_session = sshClient('172.17.110.143', '22', 'pi', 'raspberry')
except paramiko.AuthenticationException as e:
    print("paramiko.AuthenticationException: %s" % e)
    print("Can`t open ssh session at 172.17.110.5, please try again later")
    # return False
except paramiko.ssh_exception.SSHException as e:
    print("paramiko.SSHException: %s" % e)
    print("Can`t open ssh session at 172.17.110.5, please try again later")
    # return jsonify(False)

# execute remote command
print('sender')
try:
    res = sshCommand(ssh_session, "python3 send_photo.py").decode()
    print('ss', res[:100], res[-100:])
except paramiko.ssh_exception.SSHException as e:
    print(e)
    sys.exit(1)
    # return jsonify(False)
print(type(res))
image = base64_decode_image(res, np.uint8, (300, 300, 3))
# image2 = cv2.resize(image, (1080, 720))
# print(image)
cv2.imshow('ss', image)
cv2.waitKey(0)

