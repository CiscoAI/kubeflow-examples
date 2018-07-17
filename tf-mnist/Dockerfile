#This container contains your model and any helper scripts specific to your model.
FROM tensorflow/tensorflow:1.7.0

ADD mnist_model.py /opt/mnist_model.py
RUN chmod +x /opt/mnist_model.py

ENTRYPOINT ["/usr/bin/python"]
CMD ["/opt/mnist_model.py"]
