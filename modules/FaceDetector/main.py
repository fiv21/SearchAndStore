import time, os, io, math, sys, asyncio, random, cv2, argparse
import sched, logging, imutils, itertools, json, uuid, smtplib, ssl
import queue, threading
from vidgear.gears import CamGear
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, date, timedelta
from azure.iot.device.aio import IoTHubModuleClient
from azure.storage.blob import BlobClient, BlobServiceClient, ContentSettings, ContainerClient, PublicAccess
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.errors as errors
import azure.cosmos.http_constants as http_constants
from pandas.io.json import json_normalize
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


##############################
####### LAB VARIABLES ########
##############################

DEBUG_MODE = bool(os.getenv('debugMode', '')) #DEBUG_MODE: True (Laboratory / Debugging Mode ON) / False (Production Mode ON)
LOCAL_LAB = bool(os.getenv('localTestingLab', ''))  #LOCAL_LAB: True (Practia Office Lab) / False (Home Office Lab)
                                                    # CHANGE THE IP IN THE LOCAL_LAB SECTION TO MAKE THIS WORK
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


connection_string=os.getenv('local_connection_string', '')
container=os.getenv('local_container', '')

scheduleKey = os.getenv('scheduleKey', '')
scheduleUrl =  os.getenv('scheduleUrl', '')

databaseIDCosmosDB = os.getenv('databaseIDCosmosDB', '')
containerIDCosmosDB = os.getenv('containerIDCosmosDB', '')

DEVICEID = str(os.environ["IOTEDGE_DEVICEID"])

detection_model_path = 'haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)


SMTPhostPort = os.getenv('SMTPhostPort', '')
SMTPhostAddress = os.getenv('SMTPhostAddress', '') 

sender_email = os.getenv('sender_email', '')
password = os.getenv('senderEmailPassword', '') 

num_threads = int(os.getenv('threadsToUse', '')) - 1 #We force to leave 1 thread to the Man In Charge
cameraFPS = float(os.getenv('cameraFPS', ''))
##############################################################################
# FACE DETECTOR PARAMETERS #
minNeighborsParam = 3
scaleFactorParam = 1.1
############################
# GET DATETIME DATA #
today = date.today().strftime("%d/%m/%Y")
now = datetime.strptime(str(today + " " + datetime.now().strftime("%H:%M")), "%d/%m/%Y %H:%M")
refreshSchedule = datetime.now()
FPS = 0.5
timeoutFlag = False
stopWatch = datetime.now()
fpsCounter = 0
#####################
# GLOBAL DISGUSTING PARAMETERS #
lastNombreProfesor = "NAME"
lastMailProfesor = "NAME@DOMAIN.DOT"
lastNombreCurso = "COURSENAME"
ProgramID = "999"
MatterID = "999"
LessonID = "999"
InstitutionID = "Practia Global"
###################################
### PROCESS QUEUES DEFINITIONS ####
###################################
q = queue.Queue()
#####################
### RTSP Objects ####
#####################
options4k = {"CAP_PROP_FRAME_WIDTH ":3840, "CAP_PROP_FRAME_HEIGHT":2160, "CAP_PROP_FPS ":cameraFPS}
optionsHD = {"CAP_PROP_FRAME_WIDTH ":1920, "CAP_PROP_FRAME_HEIGHT":1080, "CAP_PROP_FPS ":cameraFPS}
######################################################################################################
if LOCAL_LAB:
    DEBUG_CAM = str('rtsp://practia:global@192.168.1.110:5554/video/h264')
else:
    DEBUG_CAM = str('rtsp://practia:global@192.168.88.13:5554/video/h264')

if DEBUG_MODE:
    logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.DEBUG)
else:
    logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO)
######################################################################################################


if DEBUG_MODE:
    logging.debug('CREATING LABORATORY CAMERA STREAMING...')
    try:
        testingCamera = CamGear(source=DEBUG_CAM, logging = True, **optionsHD).start()
    except RuntimeError:
        logging.error('DEBUG CAMERA OFFLINE')
        pass
else:
    logging.warning('Establishing connection with RTSP cameras...')
    try:
        camera1 = CamGear(source=RTSP_cam1, logging = True, **options4k).start()
    except RuntimeError:
        logging.error('RTSP CAMERA {} OFFLINE'.format(RTSP_cam1))
        pass
    try:
        camera2 = CamGear(source=RTSP_cam2, logging = True, **options4k).start()
    except RuntimeError:
        logging.error('RTSP CAMERA {} OFFLINE'.format(RTSP_cam2))
        pass



def worker(a): #The guys who will work while the man in charge supervise the operation
    while True:
        f, args = q.get()
        f(*args)
        q.task_done()

threads = []

for i in range(3): #We create the workers who will do the hard work ASAP
    w = threading.Thread(target=worker, args=(q,))
    w.setDaemon(True)
    w.start()
    threads.append(w)



def notifyProfessor(nombreProfesor, mailProfesor, nombreCurso, inicio):
    message = MIMEMultipart("alternative")
    if inicio == True:
        html = """<html><head></head><body>\
    <p><h1>Notificacion de inicio de analisis de clase</h1><br>""" + str(nombreProfesor) +""": la clase de\
         """+nombreCurso+ """ ha comenzado<br>Ante cualquier consulta no dude en contactarnos, \
        desde <a href="practia.global">Practia Global</a>.
        </p>
    </body>
    </html>
    """
    else:
        html = """<html><head></head><body>\
    <p><h1>Notificacion de Fin de analisis de clase</h1><br>""" + str(nombreProfesor) +""": la clase de\
         """+nombreCurso+ """ ha finalizado<br>Ante cualquier consulta no dude en contactarnos, \
        desde <a href="practia.global">Practia Global</a>.
        </p>
    </body>
    </html>
    """

    messageBody = MIMEText(html, "html")
    message.attach(messageBody)
    mailserver = smtplib.SMTP(SMTPhostAddress, SMTPhostPort)
    mailserver.ehlo()
    mailserver.starttls()
    mailserver.login(sender_email, password)
    mailserver.sendmail(sender_email, mailProfesor, message.as_string())
    mailserver.quit()

def checkSchedule(status):
    global lastNombreProfesor
    global lastMailProfesor
    global lastNombreCurso
    global ProgramID
    global MatterID
    global LessonID
    global InstitutionID
    logging.info('Checking the schedule')
    today = date.today().strftime("%d/%m/%Y")
    now = datetime.strptime(str(today + " " + datetime.now().strftime("%H:%M")), '%d/%m/%Y %H:%M')
    getLastUpdate = "SELECT * FROM "+ str(containerIDCosmosDB) + " s WHERE s.edgeDeviceUID = '"+ str(DEVICEID) +"' ORDER BY s._ts DESC"
    state = False
    finClase = now - timedelta(minutes=10)
    timeoutInMinutes = 1
    FPS = 1
    FEEDOPTIONS = {}
    FEEDOPTIONS["enableCrossPartitionQuery"] = True
    QUERY = {
        "query": getLastUpdate
    }
    client = cosmos_client.CosmosClient(str(scheduleUrl), {'masterKey': str(scheduleKey)})
    connectionStringCosmosDB = "dbs/" + str(databaseIDCosmosDB) + "/colls/" + str(containerIDCosmosDB)
    results = list(client.QueryItems(connectionStringCosmosDB, QUERY, FEEDOPTIONS))
    if len(results) > 0:
        df = json_normalize(results)
        for y in range(len(df['profesor.itinerario'])):
            for x in range(len(df["profesor.itinerario"][y])):
                if (df["profesor.itinerario"][y][x]['diaMesAnio'] == today):
                    inicioClase = datetime.strptime(str(df["profesor.itinerario"][y][x]['diaMesAnio'] + " " +
                                                        df["profesor.itinerario"][y][x]['horarioInicio']), '%d/%m/%Y %H:%M')
                    auxfinClase = datetime.strptime(str(df["profesor.itinerario"][y][x]['diaMesAnio'] + " " +
                                                        df["profesor.itinerario"][y][x]['horarioFin']), '%d/%m/%Y %H:%M')
                    timeoutInMinutes = int(df["profesor.itinerario"][y][x]['timeoutInMinutes'])
                    try:
                        ProgramID = df['profesor.itinerario'][y][x]['programaID']
                        MatterID = df['profesor.itinerario'][y][x]['cursoID']
                        LessonID = df['profesor.itinerario'][y][x]['claseID']
                        InstitutionID = df['institucion'][y]
                    except:
                        logging.error("Failure getting metadata!")
                        ProgramID = "000"
                        MatterID = "000"
                        LessonID = "000"
                        InstitutionID = "Metadata ERROR!"
                        pass
                    FPS = float(cameraFPS/float(df.fpsRate[y]))
                    if (inicioClase <= now and now < auxfinClase and status == False):
                        nombreCurso = str(df["profesor.itinerario"][y][x]['nombreCurso'])
                        nombreProfesor = str(df["profesor.nombre"][y])
                        mailProfesor = str(df['profesor.email'][y])
                        logging.debug('Sending notification via e-mail')
                        lastNombreProfesor = nombreProfesor
                        lastMailProfesor = mailProfesor
                        lastNombreCurso = nombreCurso
                        state = True
                        q.put( (notifyProfessor, [nombreProfesor, mailProfesor, nombreCurso, state]) )
                        logging.info('Class started!')
                        finClase = auxfinClase
                        return (state, finClase, timeoutInMinutes, FPS)
    return (state, finClase, timeoutInMinutes, FPS)

def faceCutter(proc, gray, countFace, captureTime):
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
    faces = face_detection.detectMultiScale(gray,scaleFactor=scaleFactorParam,minNeighbors=minNeighborsParam, minSize=(40,40),flags=cv2.CASCADE_SCALE_IMAGE)
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
    logging.debug('Putting frame in queue to persist it...')
    q.put( (storePicture, [np.array(new_im), captureTime]) )

def detectFace(rawRTSPCapture, timeToStamp, setTimeOutInMinutes):  
    global stopWatch  
    global timeoutFlag
    gray = cv2.cvtColor(rawRTSPCapture, cv2.COLOR_BGR2GRAY)
    rawRTSPCapture = cv2.cvtColor(rawRTSPCapture, cv2.COLOR_BGR2RGB)
    faces = face_detection.detectMultiScale(gray,scaleFactor=scaleFactorParam,minNeighbors=minNeighborsParam, minSize=(40,40),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        logging.debug('{} Face(s) detected!'.format(len(faces)))
        q.put( (faceCutter, [rawRTSPCapture, gray, len(faces), timeToStamp]) )
        stopWatch = datetime.now()
    else:
        logging.debug('No Face(s) detected in frame')
        auxVar = stopWatch + timedelta(minutes=setTimeOutInMinutes)
        auxVar = int(float(datetime.timestamp(auxVar))*1000)
        if (timeToStamp > auxVar):
            timeoutFlag = True
            logging.warning('CLASS TIMEDOUT!')

def storePicture(rtspCapture, ts):
    global ProgramID
    global MatterID
    global LessonID
    fileMetadata = {"InstitutionID":InstitutionID, "ProgramID":ProgramID, "MatterID":MatterID, "LessonID":LessonID}
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container)
    blobUploadedName = str(ts) + ".jpg"   
    fileName = "image.jpg"
    cv2.imwrite(fileName, rtspCapture)
    file_path_abs = "./" + fileName
    with open(file_path_abs, "rb") as data:
        try:
            container_client.upload_blob(blobUploadedName, data, content_settings=ContentSettings(content_type='image/jpg'), metadata=fileMetadata)
            logging.debug('Image Stored in Blob Storage')
        except:
            logging.error("Can't store the picture!")

def beginRecord(timeOutParameter, FPS):
    global fpsCounter
    if (fpsCounter % FPS == 0):
        try:
            frame1 = camera1.read()
        except NameError:
            frame1 = None
            pass
        try:
            frame2 = camera2.read()
        except NameError:
            frame2 = None
            pass
        try:
            debugFrame = testingCamera.read()
        except NameError:
            debugFrame = None
            pass
        now = datetime.now()
        exactTime = int(float(datetime.timestamp(now))*1000)
        if (frame1 != None) and (frame2 != None):
            logging.debug('Taking picture with 2 cameras...')
            #frame2 = cv2.resize(frame2, (1920, 1080), interpolation = cv2.INTER_AREA)
            passingPicture = np.concatenate((frame1, frame2), axis = 1)
        else:
            if (frame1 is not None):
                logging.debug('Taking picture with camera: %s', RTSP_cam1)
                passingPicture = frame1
            if (frame2 is not None):
                logging.debug('Taking picture with camera: %s', RTSP_cam2)
                passingPicture = frame2
            if (debugFrame is not None):
                logging.debug('Taking picture with camera: %s', DEBUG_CAM)
                passingPicture = debugFrame
            else:
                passingPicture = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
                logging.debug("NO CAMERAS ONLINE, PUTTING DARK PICTURE!")
        logging.debug("Sending frame to analyze if there's at least 1 face...")
        q.put( (detectFace, [passingPicture, exactTime, timeOutParameter]) )
    fpsCounter += 1

def manInCharge():
    state = False
    global lastNombreProfesor
    global lastMailProfesor
    global lastNombreCurso
    global timeoutFlag
    global fpsCounter
    while True:
        if (state == False):
            recording, horarioFinClase, timeoutInMinutes, FPS = checkSchedule(state)
            state = recording #it's True when the class start
        while ((timeoutFlag==False) and (recording == True)):
            if(horarioFinClase<datetime.now()):
                logging.info('Class has ended!')
                state = False #Class ended
                q.put( (notifyProfessor, [lastNombreProfesor, lastMailProfesor, lastNombreCurso, state]) )
                recording = state 
                fpsCounter = 0 
                break
            if(timeoutFlag):
                logging.warning('Class timeout! No face founded in picture')
                break       
            beginRecord(timeoutInMinutes, FPS)
            time.sleep(1/cameraFPS)
        if (horarioFinClase<datetime.now()):
            if timeoutFlag:
                logging.info('Timeout done, restarting process and requesting the actual schedule in 1 minute')
                timeoutFlag = False
            state = False
            recording = state
            fpsCounter = 0
        time.sleep(10)

t_manInCharge = threading.Thread(target=manInCharge)
t_manInCharge.daemon = True
t_manInCharge.start() #Start the daemon of the main task

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
            logging.info('System started...')
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
    for w in threads:
        w.join() #Wait until everybody finish their work, then close it.
    asyncio.run(main())