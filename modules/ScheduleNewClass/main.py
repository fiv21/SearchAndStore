import time
import os
import sys
import asyncio
from six.moves import input
import threading
import subprocess
from flask import Flask, render_template, session, redirect, url_for, flash, request, jsonify
from flask_wtf import FlaskForm
from wtforms import Form, FormField, StringField, PasswordField, BooleanField, SubmitField, DateField, IntegerField, FieldList
from wtforms_components import TimeField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap
import json
import pandas as pd
from pandas.io.json import json_normalize
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.errors as errors
import azure.cosmos.http_constants as http_constants

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.secret_key = 'SHH!'


scheduleKey = os.getenv('scheduleKey', '')
scheduleUrl =  os.getenv('scheduleUrl', '')

databaseIDCosmosDB = os.getenv('databaseIDCosmosDB', '')
containerIDCosmosDB = os.getenv('containerIDCosmosDB', '')

client = cosmos_client.CosmosClient(scheduleUrl, {'masterKey': scheduleKey})
connectionString = "dbs/" + databaseIDCosmosDB + "/colls/" + containerIDCosmosDB

DEVICEID = str(os.environ["IOTEDGE_DEVICEID"])

value1CamaraIP = os.getenv('value1CamaraIP', '')
value2CamaraIP = os.getenv('value2CamaraIP', '')

ip_whitelist = ['localhost', '127.0.0.1', '172.18.0.1', '10.0.80.10']


def valid_ip():
    client = request.remote_addr
    if client in ip_whitelist:
        return True
    else:
        return False



def getJSONexample():
    course = '''
    {
        "edgeDeviceUID": "IoTMakerSpaceTestDevice",
        "aula": "Maker Space",
        "institucion": "Practia Global",
        "fpsRate": "1",
        "camarasIP": {
            "MAC1": {
                "value": "00:00:00:00:00:00"
            },
            "MAC2": {
                "value": "00:00:00:00:00:00"
            }
        },
        "profesor": {
            "nombre": "Franco Vargas",
            "email": "fivargas@practia.global",
            "itinerario": []
          }
        }'''
    course = json.loads(course)
    return course

def updateCosmosDB(scheduleJSON):
   try: 
      client.CreateItem(connectionString, scheduleJSON)
      return True
   except errors.HTTPFailure as e:
        if e.status_code >= 400: # Move these constants to an enum
            return False
        else: 
            raise errors.HTTPFailure(e.status_code)

def fromMultiDicToJSON(data):
   course = getJSONexample()
   course['edgeDeviceUID'] = DEVICEID
   nombreProfesor = data.get('name')
   course['profesor']['nombre'] = nombreProfesor
   emailProfesor = data.get('email')
   course['profesor']['email'] = emailProfesor
   aulaClase = data.get('aula')
   course['aula'] = aulaClase
   fpsRate = data.get('fpsRate')
   course['fpsRate'] = fpsRate
   nombreInstitucion = data.get('institution')
   course['institucion'] = nombreInstitucion
   listaItinerario = []
   numero = 0
   parsear = 'clase-'+str(numero)
   while(data.get(str(parsear+'-cursoID'))!=None):
      programaID = data.get(str(parsear+'-programaID'))
      cursoID = data.get(str(parsear+'-cursoID'))
      nombreCurso = data.get(str(parsear+'-nombreCurso'))
      claseID = data.get(str(parsear+'-claseID'))
      diaMesAnio = data.get(str(parsear+'-date'))
      horarioInicio = data.get(str(parsear+'-horarioInicio'))
      horarioFin = data.get(str(parsear+'-horarioFin'))
      timeoutInMinutes = data.get(str(parsear+'-timeoutInMinutes'))
      cargar =  {
                "programaID": "999",
                "cursoID": "992",
                "nombreCurso": "Testing de Domingo por la noche",
                "claseID":"999",
                "diaMesAnio": "30/12/2019",
                "horarioInicio": "15:30:00",
                "horarioFin": "15:40:00",
                "timeoutInMinutes": "1"
            }
      cargar['programaID'] = programaID
      cargar['cursoID'] = cursoID
      cargar['nombreCurso'] = nombreCurso
      cargar['claseID'] = claseID
      cargar['diaMesAnio'] = diaMesAnio
      cargar['horarioInicio'] = horarioInicio
      cargar['horarioFin'] = horarioFin
      cargar['timeoutInMinutes'] = timeoutInMinutes
      listaItinerario.append(cargar)
      numero+=1
      parsear='clase-'+str(numero)
   course['profesor']['itinerario'].extend(listaItinerario)
   return course


class scheduleForm(Form):
   programaID = IntegerField('ID Programa en Curso')
   cursoID = IntegerField('ID Curso')
   nombreCurso = StringField('Nombre del Curso') #, validators=[DataRequired()])
   claseID = IntegerField('ID Clase')
   date = DateField('Fecha',format='%d-%m-%y') #, validators=[DataRequired()])
   horarioInicio = TimeField('Horario de Inicio') #, validators=[DataRequired()])
   horarioFin = TimeField('Horario de Finalizacion') #, validators=[DataRequired()])
   timeoutInMinutes = IntegerField('Tiempo de Espera')

class leForm(FlaskForm):
   name = StringField('Nombre de Profesor') #, validators=[DataRequired()])
   email = StringField('Correo Electronico') #, validators=[DataRequired()])
   aula = StringField('Aula') #, validators=[DataRequired()])
   fpsRate = StringField('Imagenes por Segundo (FPS)')
   institution = StringField('Institucion') #, validators=[DataRequired()])
   clase = FieldList(FormField(scheduleForm), min_entries=1)
   button = SubmitField()


@app.route('/success') 
def success(): 
   return 'Carga de clases guardada exitosamente!' 

@app.route('/failure') 
def failure(): 
   return 'ERROR!'

@app.route('/', methods=['POST','GET'])
def home():
   form = leForm()
   if valid_ip():
      if request.method == 'POST': 
         data = request.form
         dataJSON =  fromMultiDicToJSON(data)
         saved = updateCosmosDB(dataJSON)
         if saved:
            print("Success!")
            return redirect(url_for('success')) 
         else:
            return redirect(url_for('failure'))
      return render_template('index.html', form=form)
   else:
      return """<title>401 Unauthorized</title>
                <h1> Unauthorized </h1>
               <p>You're trying to access from a non whitelisted IP.</p>""", 401



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8086)