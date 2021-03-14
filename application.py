import os

from flask import Flask, flash, redirect, render_template, request, session

# Configure application
app = Flask(__name__)

@app.route("/")
def index():

    return "test"

if __name__ == '__main__':
    app.run()

    # Uncomment to enable HTTPS/SSL and comment out the line above
    #app.run(host='0.0.0.0',port=443,ssl_context=context)