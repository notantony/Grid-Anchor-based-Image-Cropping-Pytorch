from flask import Flask

app = Flask("cropping-server")

from server.app import routes