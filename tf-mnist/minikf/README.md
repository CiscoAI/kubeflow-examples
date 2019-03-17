# Table of Contents
- [Overview of the application in miniKF](#overview-of-the-application)
- [Prerequisites](#prerequisites)
- [MiniKF VM](#login-to-minikf-vm)
- [Steps in MiniKF VM](#steps-in-vm)
	- [Installation](#installation)
	- [Setup](#setup)
	- [Model Testing](#model-testing)
	- [Extras](#extras)

# Overview of the application
This tutorial contains instructions to build an **end to end kubeflow app** on a
Kubernetes cluster running on miniKF with minimal prerequisites.
The mnist model is trained and served from an NFS mount.
**This example is intended for
beginners with zero/minimal experience in kubeflow.**

This tutorial demonstrates:

* Train a simple MNIST Tensorflow model (*See mnist_model.py*)
* Export the trained Tensorflow model and serve using tensorflow-model-server
* Test/Predict images with a python client(*See mnist_client.py*)




# Prerequisites

## MiniKF

Install miniKF by following the instructions outlined here(https://www.kubeflow.org/docs/started/getting-started-minikf/)
If the IP, 10.10.10.10 is accessible from the broswer, you are good to go !

# Login to MiniKF VM

	
	vagrant ssh
	
	
# Steps in VM

Install the following packages
	
	sudo apt-get install nfs-common
	
Clone the KFLab repo
	
	git clone https://github.com/ciscoAI/KFLab YOUR_REPO_DIR
	cd YOUR_REPO_DIR/tf-mnist/minikf 
	
## Installation

        ./install.bash
	# Ensure that all pods are running in the namespace set in variables.bash. The default namespace is kubeflow
        kubectl get pods -n kubeflow

If there is any rate limit error from github, please follow the instructions at:
[Github Token Setup](https://github.com/ksonnet/ksonnet/blob/master/docs/troubleshooting.md#github-rate-limiting-errors)


## Setup

1.  (**Optional**) If you want to use a custom image for training, create the training Image and upload to DockerHub. Else, skip this step to use the already existing image (`gcr.io/cpsg-ai-demo/tf-mnist-demo:v1`).

   Point `DOCKER_BASE_URL` to your DockerHub account. Point `IMAGE` to your training image. If you don't have a DockerHub account,
   create one at [https://hub.docker.com/](https://hub.docker.com/), upload your image there, and do the following
   (replace <username> and <container> with appropriate values).

       DOCKER_BASE_URL=docker.io/<username>
       IMAGE=${DOCKER_BASE_URL}/<image>
       docker build . --no-cache  -f Dockerfile -t ${IMAGE}
       docker push ${IMAGE}


2. Run the training job setup script

	   ./train.bash
       # Ensure that all pods are running in the namespace set in variables.bash. The default namespace is kubeflow
       kubectl get pods -n kubeflow


3. Start TF serving on the trained results

       ./serve.bash


4. Port forward to access the serving port locally

    	./portf.bash
    
    
## Model Testing

The model can be tested using a python client or via web application

### Using a python client from the laptop


Run a sample client code to predict images(See mnist-client.py)

    virtualenv --system-site-packages env
    source ./env/bin/activate
    easy_install -U pip
    pip install --upgrade tensorflow
    pip install tensorflow-serving-api
    pip install python-mnist
    pip install Pillow

    TF_MODEL_SERVER_HOST=10.10.10.10 TF_MNIST_IMAGE_PATH=data/7.png python mnist_client.py

You should see the following result

    Your model says the above number is... 7!

Now try a different image in `data` directory :)

### Using a web application
#### NodePort

Another way to expose your web application on the Internet is NodePort. Define
variables in variables.bash and run the following script:

 ```
   ./webapp.bash
 ```

After running this script, you will get the IP adress of your web application.
Open browser and see app at http://10.10.10.10:NodePort from your laptop.

## Extras

## Retrain your model

If you want to change the training image, set `image` to your new training
image. See the [prototype
generation](https://github.com/CiscoAI/kubeflow-workflows/blob/d6d002f674c2201ec449ebd1e1d28fb335a64d1e/mnist/train.bash#L21)

        ks param set ${JOB} image ${IMAGE}

If you would like to retrain the model(with a new image or not), you can delete
the current training job and create a new one. See the
[training](https://github.com/CiscoAI/kubeflow-workflows/blob/d6d002f674c2201ec449ebd1e1d28fb335a64d1e/mnist/train.bash#L28)
step.

         ks delete ${KF_ENV} -c ${JOB}
         ks apply ${KF_ENV} -c ${JOB}

### Clean up pods
	
	./cleanup.bash

   Forcefully terminate pods using:
   
   	$ kubectl delete pod <pod_name> --force -n kubeflow --grace-period=0

    
