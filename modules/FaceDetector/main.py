import time
import logging
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
from datetime import datetime, date, timedelta
from scipy.linalg import norm
from scipy import sum, average
import itertools
from azure.iot.device.aio import IoTHubModuleClient
from azure.storage.blob import BlobClient, BlobServiceClient, ContentSettings, ContainerClient, PublicAccess
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.errors as errors
import azure.cosmos.http_constants as http_constants
import json
import pandas as pd
from pandas.io.json import json_normalize
import uuid
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


logging.basicConfig(format="%(asctime)s - %(levelname)s:-8s%(message)s", level=logging.INFO)

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
RTSP_cam2 = str('rtsp://'+rtspUser2+':'+rtspPass2+'@'+IPCam2+':'+PortCam2+PathCam2)

connection_string=os.getenv('connection_string', '')
container=os.getenv('container', '')

scheduleKey = os.getenv('scheduleKey', '')
scheduleUrl =  os.getenv('scheduleUrl', '')

databaseIDCosmosDB = os.getenv('databaseIDCosmosDB', '')
containerIDCosmosDB = os.getenv('containerIDCosmosDB', '')

DEVICEID = str(os.environ["IOTEDGE_DEVICEID"])

detection_model_path = 'haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)
##############################################################################

# FACE DETECTOR PARAMETERS #
minNeighborsParam = 5
scaleFactorParam = 1.3
############################

# GET DATETIME DATA #
today = date.today().strftime("%d/%m/%Y")
now = datetime.strptime(str(today + " " + datetime.now().strftime("%H:%M:%S")), "%d/%m/%Y %H:%M:%S")
refreshSchedule = datetime.now()
delay = 4
#####################


def notifyProfessor():




def checkSchedule():
    logging.info('Checking the schedule')
    today = date.today().strftime("%d/%m/%Y")
    now = datetime.strptime(str(today + " " + datetime.now().strftime("%H:%M:%S")), '%d/%m/%Y %H:%M:%S')
    getLastUpdate = "SELECT TOP 1 * FROM "+ str(containerIDCosmosDB) + " s WHERE s.edgeDeviceUID = '"+ str(DEVICEID) +"' ORDER BY s._ts DESC"
    FEEDOPTIONS = {}
    FEEDOPTIONS["enableCrossPartitionQuery"] = True
    QUERY = {
        "query": getLastUpdate
    }
    client = cosmos_client.CosmosClient(str(scheduleUrl), {'masterKey': str(scheduleKey)})
    connectionStringCosmosDB = "dbs/" + str(databaseIDCosmosDB) + "/colls/" + str(containerIDCosmosDB)
    results = list(client.QueryItems(connectionStringCosmosDB, QUERY, FEEDOPTIONS))
    df = json_normalize(results)
    for y in range(len(df['profesor.itinerario'])):
        for x in range(len(df["profesor.itinerario"][y])):
            if (df["profesor.itinerario"][y][x]['diaMesAnio'] == today):
                inicioClase = datetime.strptime(str(df["profesor.itinerario"][y][x]['diaMesAnio'] + " " +
                                                    df["profesor.itinerario"][y][x]['horarioInicio']), '%d/%m/%Y %H:%M:%S')
                finClase = datetime.strptime(str(df["profesor.itinerario"][y][x]['diaMesAnio'] + " " +
                                                    df["profesor.itinerario"][y][x]['horarioFin']), '%d/%m/%Y %H:%M:%S')
                timeoutInMinutes = int(df["profesor.itinerario"][y][x]['timeoutInMinutes'])
                delay = (1.0/int(df.fpsRate[y]))
                if (inicioClase <= now and now <= finClase):
                    nombreCurso = str(df["profesor.itinerario"][y][x]['nombreCurso'])
                    nombreProfesor = str(df["profesor.nombre"])
                    notifyProfessor()
                    state = True
                    logging.info('Class started!')
            else:
                state = False
                finClase = now
                timeoutInMinutes = 1
                delay = 4
            return (state, finClase, timeoutInMinutes, delay)


def faceCutter(proc, gray, countFace):
    gridX = 7
    gridY= 4
    zero = "0"  
    stepT = int(proc.shape[0]/gridY) 
    stepL = int(proc.shape[1]/gridX) 
    m_x = 10
    m_y = 10
    img_w = 200
    img_h = 300
    matrix = [ [ zero for i in range(gridX) ] for j in range(gridY) ]
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
    return 1

def detectFace(rawRTSPCapture):    
    gray = cv2.cvtColor(rawRTSPCapture, cv2.COLOR_BGR2GRAY)
    rawRTSPCapture = cv2.cvtColor(rawRTSPCapture, cv2.COLOR_BGR2RGB)
    faces = face_detection.detectMultiScale(gray,scaleFactor=scaleFactorParam,minNeighbors=minNeighborsParam,
                                            minSize=(40,40),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        print("Faces detected: {}".format(len(faces)))
        faceCutter(rawRTSPCapture, gray, len(faces))
        return 1
    else:
        return 0

def storePicture(rtspCapture):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container)
    now = datetime.now()
    ts = str(int(datetime.timestamp(now)))
    fileName = "image.jpg"
    cv2.imwrite(fileName, rtspCapture)
    file_path_abs = "./" + fileName
    blobUploadedName = ts + ".jpg"
    with open(file_path_abs, "rb") as data:
        try:
            container_client.upload_blob(blobUploadedName, data, content_settings=ContentSettings(content_type='image/jpg'))
        except:
            logging.error('STORING picture!')
            time.sleep(1)
            return 0
    return 1


def beginRecord():
    try:
        camera1 = cv2.VideoCapture(RTSP_cam1)
        camera2 = cv2.VideoCapture(RTSP_cam2)
    except:
        logging.critical('Exception error: opening stream over RTSP!')
        logging.debug('Restarting in 10 seconds...')
        time.sleep(10)
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()
    if ret1 and ret2:
        logging.debug('Taking picture with 2 cameras...')
        frame2 = cv2.resize(frame2, (1920, 1080), interpolation = cv2.INTER_AREA)
        bigPicture = np.concatenate((frame1, frame2), axis = 1)
        detectTimeout = detectFace(bigPicture)
    else:
        if ret1:
            logging.debug('Taking picture with camera: %s', RTSP_cam1)
            detectTimeout = detectFace(frame1)
        if ret2:
            logging.debug('Taking picture with camera: %s', RTSP_cam2)
            detectTimeout = detectFace(frame2)
    if detectTimeout == 0:
        return 0
    try:
        camera1.release()
        camera2.release()
    except:
        logging.error("Can't release the stream channel")
    time.sleep(1)
    return 1

async def main():
    try:
        if not sys.version >= "3.5.3":
            raise Exception( "The sample requires python 3.5.3+. Current version of Python: %s" % sys.version )
        logging.info('IoT Hub Client for Python')
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
            timeoutFlag = False
            counterTimeout = 0
            logging.info('System starting...')
            while True:
                state, horarioFinClase, timeoutInMinutes, delay = checkSchedule()
                while (timeoutFlag==False and state == True):
                    while(beginRecord() == 0):
                        counterTimeout+=1
                        time.sleep(delay)
                        if counterTimeout == (timeoutInMinutes*60):
                            logging.error('NO FACES DETECTED, MODULE TIMEDOUT! WAIT UNTIL NEXT CLASS')
                            timeoutFlag = True
                            break                      
                if ((horarioFinClase-datetime.now()) == 0):
                    if timeoutFlag:
                        logging.info('Timeout done, restarting process and requesting the actual schedule in 1 minute')
                    timeoutFlag = False
                    counterTimeout = 0
                time.sleep(60)




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