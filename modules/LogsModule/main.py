import time
import os
import sys
import asyncio
from six.moves import input
import threading
from azure.iot.device.aio import IoTHubModuleClient
from flask import Flask
from flask import request
import subprocess


app = Flask(__name__)
ip_whitelist = ['localhost', '127.0.0.1', '172.18.0.1', '10.0.80.10', '192.168.1.165', '192.168.88.30', '192.168.88.4']


DEVICEID = str(os.environ["IOTEDGE_DEVICEID"])


def valid_ip():
    client = request.remote_addr
    if client in ip_whitelist:
        return True
    else:
        return False

@app.route("/")
def hello():
  return "Go to localhost:8080/status/"

@app.route('/status/')
def get_status():
    if valid_ip():
        command_check = "ls"
        command_modules = "ls -a"
        command_date = "date"

        try:
            result_check = subprocess.check_output(
                [command_check], shell=True)
            result_modules = subprocess.check_output(
                [command_modules], shell=True)
            result_date = subprocess.check_output(
                [command_date], shell=True)
        except subprocess.CalledProcessError as e:
            return "An error occurred while trying to fetch task status updates."

        return """<b><h1>IoT Edge module List:</h1></b><br> <pre>%s</pre>, <br><br> <b><h1>IoT Edge:</h1></b>  <br> <pre>%s</pre>, <br><br> <b><h1>Date:</h1></b> <pre>%s</pre>""" % (result_check, result_modules, result_date)
    else:
        return """<title>401 Unauthorized</title>
                <h1> Unauthorized </h1>
               <p>You're trying to access from a non whitelisted IP.</p>""", 401


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
