# Overview of the application
This tutorial contains instructions to build an **end to end kubeflow app** on a
native Kubernetes cluster(non-cloud) with minimal prerequisites.  The mnist
model is trained and served from a NFS mount.  This example is intended for
beginners with zero/minimal experience in kubeflow.

This tutorial demonstrates:

* Train a simple MNIST Tensorflow model (*See mnist_model.py*)
* Export the trained Tensorflow model and serve using tensorflow-model-server
* Test/Predict images with a python client(*See mnist_client.py*)

## Overall Structure
![Generic Schematic](pictures/generic_schematic.png?raw=true "Generic Schematic of MNIST application")
This picture shows the overall schematic without any Google Kubernetes Engine
specifics. After the `install` step, the MNIST images are downloaded (from
inside the code) to the `train` stage. The `train` stage keeps updating the
parameters in the `persistent parameter storage`, a storage that persists even
if the containers and the clusters go down. The `client` app sends an image to
the `serve` stage that in turn retrieves the parameters from the `persistent
parameter storage` and uses these parameters to predict the image and send
the response back to the `client`.

## GKE Specific Structure
![Google Kubernetes Engine Schematic](pictures/gke_schematic.png?raw=true "GKE Schematic of MNIST application")
The exact implementation on GKE looks somewhat like the above image. There is
one `tf-master` and possibly multiple `tf-worker` and `tf-ps` (parameter server)
pods that form the `train` stage.
The exact number of replicas for each pod is outside the scope of this document and
_is a hyperparameter to be played with in order to scale this application_.
The parameter are then stored on an `NFS` persistent volume. Finally, multiple
`tf-serve` pods implement the `serve` stage. The `Browser/App` (outside the
logical GKE cluster where the `train` and `serve` stages run) connects with the
`serve` stage to get a prediction of an image.


# Prerequisites

1. **kubectl cli**

   Check if kubectl  is configured properly by accessing the cluster Info of your kubernetes cluster

       $kubectl cluster-info

 2. **Ksonnet**(Latest version:version: 0.11.0)

    Check ksonnet version

        $ ks version


If above commands succeeds, you are good to go !


# Installation

        ./install.bash

If there is any rate limit error from github, please follow the instructions at:
https://ksonnet.io/docs/tutorial#troubleshooting-github-rate-limiting-errors.

# Setup
1. Create the mnist training Image and upload to DockerHub

   Point `DOCKER_BASE_URL` to your DockerHub account. If you don't have a DockerHub account, create one.

       DOCKER_BASE_URL=docker.io/johnugeorge
       IMAGE=${DOCKER_BASE_URL}/tfmodel
       docker build . --no-cache  -f Dockerfile -t ${IMAGE}
       docker push ${IMAGE}


2. Run the training job setup script

       ./train.bash

3. Start TF serving on the trained results

       ./serve.bash

# Model Testing

The model can be tested using a local python client or via web application

## Using a local python client

 Port forward to access the serving port locally


       SERVING_POD_NAME=`kubectl -n ${NAMESPACE} get pod -l=app=mnist -o jsonpath='{.items[0].metadata.name}'`

       kubectl -n ${NAMESPACE} port-forward ${SERVING_POD_NAME} 9000:9000 &


 Run a sample client code to predict images(See mnist-client.py)


       virtualenv --system-site-packages env
       source ./env/bin/activate
       easy_install -U pip
       pip install --upgrade tensorflow
       pip install tensorflow-serving-api
       pip install python-mnist

       TF_MNIST_IMAGE_PATH=data/7.png python mnist_client.py

 You should see the following result

      Your model says the above number is... 7!

 Now try a different image in `data` directory :)

## Using a web application

       MNIST_SERVING_IP=`kubectl -n ${NAMESPACE} get svc/mnist --output=jsonpath={.spec.clusterIP}`
       echo "MNIST_SERVING_IP is ${MNIST_SERVING_IP}"

 Create image using Dockerfile in the webapp folder and upload to DockerHub
     
      CLIENT_IMAGE=${DOCKER_BASE_URL}/mnist-client
      docker build . --no-cache  -f Dockerfile -t ${CLIENT_IMAGE}
      docker push ${CLIENT_IMAGE}

      echo "CLIENT_IMAGE is ${CLIENT_IMAGE}"
      ks generate tf-mnist-client tf-mnist-client --mnist_serving_ip=${MNIST_SERVING_IP} --image=${CLIENT_IMAGE}

      ks apply {KF_ENV} -c tf-mnist-client
 
  Now get the loadbalancer IP of the tf-mnist-client service
 
      kubectl get svc/tf-mnist-client -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

  Open browser and see app at http://LoadBalancerIP 
