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
ip_whitelist = ['localhost', '127.0.0.1']



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
        command_check = "iotedge check"
        command_modules = "iotedge FaceDetector"
        command_docker = "docker ps -a"

        try:
            result_check = subprocess.check_output(
                [command_check], shell=True)
            result_modules = subprocess.check_output(
                [command_modules], shell=True)
            result_docker = subprocess.check_output(
                [command_docker], shell=True)
        except subprocess.CalledProcessError as e:
            return "An error occurred while trying to fetch task status updates."

        return 'check %s, modules %s, docker %s' % (result_check, result_modules, result_docker)
    else:
        return """<title>404 Not Found</title>
               <h1>Not Found</h1>
               <p>The requested URL was not found on the server.
               If you entered the URL manually please check your
               spelling and try again.</p>""", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
