import time
import os
import sys
import asyncio
from six.moves import input
import threading
import random
import cv2
import numpy as np
import argparse
import imutils
from datetime import datetime
from scipy.linalg import norm
from scipy import sum, average
from azure.iot.device.aio import IoTHubModuleClient
from azure.storage.blob import BlobClient, BlobServiceClient, ContentSettings, ContainerClient, PublicAccess
from keras.models import load_model
from keras.preprocessing.image import img_to_array


#codecs h264
#RTSP_cam1 = "rtsp://flanders:flanders123@192.168.1.125:443/videoMain"   #Foscam FHD
#RTSP_cam2 = "rtsp://admin:admin@192.168.1.115:554/video/h264"  #Cygnus 4K


############################################################################
############################################################################
rtspUser1 = 'flanders'
rtspPass1 = 'flanders123'
IPCam1 = '192.168.1.125'
PortCam1 = 443
PathCam1 = '/videoMain'

rtspUser2 = 'admin'
rtspPass2 = 'admin'
IPCam2 = '192.168.1.115'
PortCam2 = 554
PathCam2 = '/video/h264'

RTSP_cam1 = str('rtsp://'+rtspUser1+':'+rtspPass1+'@'+IPCam1+':'+str(PortCam1)+PathCam1)
RTSP_cam2 = str('rtsp://'+rtspUser2+':'+rtspPass2+'@'+IPCam2+':'+str(PortCam2)+PathCam2)
##############################################################################


connection_string='DefaultEndpointsProtocol=https;AccountName=iaestoragedev;AccountKey=HACAChorM1ugThf0VsIEYnWFiYovsTQbnOxNUrClHJaa6+++h41nixS/8QYdqRKFH1A9YoONsEB3DGjy9/IiLw==;EndpointSuffix=core.windows.net'
container='image-frame'

detection_model_path = 'haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)

def detectFace(rawRTSPCapture):
    #frameResized = imutils.resize(rawRTSPCapture,width=800)
    
    gray = cv2.cvtColor(rawRTSPCapture, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(40,40),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        print("Faces detected: {}".format(len(faces)))
        storePicture(rawRTSPCapture)
    else:
        takePicture()

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

def beginRecord():
    try:
        camera1 = cv2.VideoCapture(RTSP_cam1)
        camera2 = cv2.VideoCapture(RTSP_cam2)
    except:
        print("Exception error: opening stream over RTSP!")
        print("Restarting in 10 seconds...")
        time.sleep(10)
    while True:
        try:
            frame1 = camera1.read()[1]
            frame2 = camera2.read()[1]
        except:
            print("Exception error: taking the frame")
        try:
            frame2 = cv2.resize(frame2, (1920,1080), interpolation = cv2.INTER_AREA)
            bigPicture = np.concatenate((frame1, frame2), axis = 0)
            detectFace(bigPicture)
            time.sleep(10) #Take a picture every 3 seconds
        except:
            print("Exception error: can't handle the big frame")
            time.sleep(10)
    try:
        camera1.release()
        camera2.release()
    except:
        print("Exception error: can't release the stream channel")


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
                try:
                    beginRecord()
                except:
                    print("Can't start. Waiting 1 second before restart...")
                    time.sleep(1)


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
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

    # If using Python 3.7 or above, you can use following code instead:
    # asyncio.run(main())