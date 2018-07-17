import os
import random
import numpy
import logging
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from PIL import Image
from grpc.beta import implementations
from mnist import MNIST
from flask import Flask, render_template, request, jsonify

app = Flask(__name__,static_url_path='/static', static_folder='static')
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)

TF_MODEL_SERVER_HOST = os.getenv("TF_MODEL_SERVER_HOST", "127.0.0.1")
TF_MODEL_SERVER_PORT = int(os.getenv("TF_MODEL_SERVER_PORT", 9000))

@app.route("/")
def main():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():

    from flask import request
    if request.method == "POST":
        # get url
        image_url = request.form.get('img')
        image_name = image_url
        app.logger.debug(image_name)
        raw_image = Image.open(image_name)
        int_image = numpy.array(raw_image)
        image = numpy.reshape(int_image, 784).astype(numpy.float32)
        channel = implementations.insecure_channel(TF_MODEL_SERVER_HOST, TF_MODEL_SERVER_PORT)
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = "mnist"
        request.model_spec.signature_name = "serving_default"
        request.inputs['x'].CopyFrom(tf.contrib.util.make_tensor_proto(image, shape=[1, 28, 28]))

        result = stub.Predict(request, 10.0)  # 10 secs timeout
        #print(result)
        #print(MNIST.display(image, threshold=0))
        app.logger.debug("Your model says the above number is... %d!" % result.outputs["classes"].int_val[0])
        return str(result.outputs["classes"].int_val[0])
        #return "1"

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)
