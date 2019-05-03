#
# This container contains your model and any helper scripts specific to your model.
# In case you modify the training code, docker has to be run with this file to
# regenerate the docker image
#
FROM tensorflow/tensorflow:1.7.0

ADD mnist_model.py /opt/mnist_model.py
RUN chmod +x /opt/mnist_model.py

ENTRYPOINT ["/usr/bin/python"]
CMD ["/opt/mnist_model.py"]
