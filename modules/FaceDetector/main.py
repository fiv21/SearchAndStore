import time
import os
import io
import math
import sys
import asyncio
from six.moves import input
import threading
import random
import cv2
import numpy as np
import argparse
import imutils
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from scipy.linalg import norm
from scipy import sum, average
import itertools
from azure.iot.device.aio import IoTHubModuleClient
from azure.storage.blob import BlobClient, BlobServiceClient, ContentSettings, ContainerClient, PublicAccess
from keras.models import load_model
from keras.preprocessing.image import img_to_array

###############################################################################
rtspUser1 = os.getenv('rtspUser1', '')
rtspPass1 = os.getenv('rtspPass1', '')
IPCam1 = os.getenv('IPCam1', '')
PortCam1 = os.getenv('PortCam1', '')
PathCam1 = os.getenv('PathCam1', '')

rtspUser2 = os.getenv('rtspUser2', '')
rtspPass2 = os.getenv('rtspPass2', '')
IPCam2 = os.getenv('IPCam2', '')
PortCam2 = os.getenv('PortCam2', '')
PathCam2 = os.getenv('PathCam2', '')

RTSP_cam1 = str('rtsp://'+rtspUser1+':'+rtspPass1+'@'+IPCam1+':'+PortCam1+PathCam1)
#RTSP_cam1 = str('http://192.168.1.125:443/cgi-bin/CGIProxy.fcgi?cmd=snapPicture2&usr=flanders&pwd=flanders123')
RTSP_cam2 = str('rtsp://'+rtspUser2+':'+rtspPass2+'@'+IPCam2+':'+PortCam2+PathCam2)

connection_string=os.getenv('connection_string', '')
container=os.getenv('container', '')

detection_model_path = 'haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)
##############################################################################

minNeighborsParam = 5
scaleFactorParam = 1.3

def faceCutter(proc, gray, countFace):
    gridX = 7
    gridY= 4
    zero = "0"  
    matrix = [ [ zero for i in range(gridX) ] for j in range(gridY) ]
    stepT = int(proc.shape[0]/gridY) 
    stepL = int(proc.shape[1]/gridX) 
    m_x = 10
    m_y = 10
    img_w = 200
    img_h = 300
    i = 0
    j = 0
    new_im_w = gridX*img_w+m_x*gridX
    new_im_h = gridY*img_h+m_y*gridY
    new_im = Image.new('RGB', (new_im_w, new_im_h))
    faces = face_detection.detectMultiScale(gray,scaleFactor=scaleFactorParam,minNeighbors=minNeighborsParam,
                                            minSize=(40,40),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        croppedFace = proc[y:y+h, x:x+w]
        croppedFace = cv2.resize(croppedFace, (img_w, img_h), interpolation = cv2.INTER_AREA)
        croppedFace = Image.fromarray(croppedFace)
        auxAbsDist = 200000
        auxL = 0 
        auxK = 0
        for k in range(0, gridY):
            for l in range(0, gridX):
                if matrix[k][l]==zero:
                    relativeDistance = math.sqrt((stepT*k-(y+h*0.5))**2+(l*stepL-(x+w*0.5))**2)
                    if(relativeDistance<auxAbsDist):
                        auxL = l
                        auxK = k
                        candidate = (l*(img_w+m_x), (img_h+m_y)*k)
                        auxAbsDist = relativeDistance
        matrix[auxK][auxL] = 'busy'
        new_im.paste(croppedFace, candidate)
    new_im.show()
    storePicture(np.array(new_im))
    return 0

def detectFace(rawRTSPCapture):    
    gray = cv2.cvtColor(rawRTSPCapture, cv2.COLOR_BGR2GRAY)
    rawRTSPCapture = cv2.cvtColor(rawRTSPCapture, cv2.COLOR_BGR2RGB)
    faces = face_detection.detectMultiScale(gray,scaleFactor=scaleFactorParam,minNeighbors=minNeighborsParam,
                                            minSize=(40,40),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        print("Faces detected: {}".format(len(faces)))
        faceCutter(rawRTSPCapture, gray, len(faces))
    else:
        return 0

def storePicture(rtspCapture):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container)
    now = datetime.now()
    ts = str(datetime.timestamp(now))
    fileName = "image.jpg"
    cv2.imwrite(fileName, rtspCapture)
    file_path_abs = "./" + fileName
    blobUploadedName = ts + ".jpg"
    with open(file_path_abs, "rb") as data:
        try:
            container_client.upload_blob(blobUploadedName, data, content_settings=ContentSettings(content_type='image/jpg'))
        except:
            print("Exception error: STORING picture!")
            time.sleep(1)
    return 0


def beginRecord():
    try:
        camera1 = cv2.VideoCapture(RTSP_cam1)
        camera2 = cv2.VideoCapture(RTSP_cam2)
    except:
        print("Exception error: opening stream over RTSP!")
        print("Restarting in 10 seconds...")
        time.sleep(10)
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()
    if ret1 and ret2:
        frame2 = cv2.resize(frame2, (1920, 1080), interpolation = cv2.INTER_AREA)
        bigPicture = np.concatenate((frame1, frame2), axis = 1)
        detectFace(bigPicture)
    else:
        if ret1:
            detectFace(frame1)
        if ret2:
            detectFace(frame2)
    time.sleep(1) #Take a picture every second
    try:
        camera1.release()
        camera2.release()
    except:
        print("Exception error: can't release the stream channel")
    time.sleep(1)
    return 0

async def main():
    try:
        if not sys.version >= "3.5.3":
            raise Exception( "The sample requires python 3.5.3+. Current version of Python: %s" % sys.version )
        print ( "IoT Hub Client for Python" )

        # The client object is used to interact with your Azure IoT hub.
        module_client = IoTHubModuleClient.create_from_edge_environment()

        # connect the client.
        await module_client.connect()

        # define behavior for receiving an input message on input1
        async def input1_listener(module_client):
            while True:
                input_message = await module_client.receive_message_on_input("input1")  # blocking call
                print("the data in the message received on input1 was ")
                print(input_message.data)
                print("custom properties are")
                print(input_message.custom_properties)
                print("forwarding mesage to output1")
                await module_client.send_message_to_output(input_message, "output1")

        # define behavior for halting the application


        ###everything goes HERE
        def stdin_listener():
            while True:
                beginRecord()



###and don't change much more... I'm watching you -.-"

        # Schedule task for C2D Listener
        listeners = asyncio.gather(input1_listener(module_client))

        print ( "The sample is now waiting for messages. ")

        # Run the stdin listener in the event loop
        loop = asyncio.get_event_loop()
        user_finished = loop.run_in_executor(None, stdin_listener)

        # Wait for user to indicate they are done listening for messages
        await user_finished

        # Cancel listening
        listeners.cancel()

        # Finally, disconnect
        await module_client.disconnect()

    except Exception as e:
        print ( "Unexpected error %s " % e )
        raise

if __name__ == "__main__":
    #loop = asyncio.get_event_loop()
    #loop.run_until_complete(main())
    #loop.close()

    # If using Python 3.7 or above, you can use following code instead:
     asyncio.run(main())