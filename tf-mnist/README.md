# Table of Contents
- [What you will learn](#what-you-will-learn)
- [Overview of the application](#overview-of-the-application)
    - [Overall Structure](#overall-structure)
    - [GKE Specific Structure](#gke-specific-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setup](#setup)
- [Model Testing](#model-testing)
    - [Using a local Python client](#using-a-local-python-client)
    - [Using a web application](#using-a-web-application)
- [Extras](#extras)

# What you will learn
Once you do complete this application you will have had hands on experience
on the following:

* How to setup a [Kubernetes](https://kubernetes.io/) cluster on [Google Cloud Platform (GCP)](https://cloud.google.com/).
* How to install [Kubeflow](https://www.kubeflow.org/) on a Kubernetes cluster.
* How to train a [TensorFlow](https://www.tensorflow.org/) model on the [MNIST](http://yann.lecun.com/exdb/mnist/index.html) data.
* How to serve the trained model using Kubeflow.
* How to develop a simple browser based application that can use the model to
  recognize handwritten digits.

# Overview of the application
This tutorial contains instructions to build an **end to end kubeflow app** on a
Kubernetes cluster running on Google Kubernetes Engine (GKE) with minimal prerequisites.
*It should work on any other K8s cluster as well.*
The mnist model is trained and served from an NFS mount.
**This example is intended for
beginners with zero/minimal experience in kubeflow.**

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

## Note for Non-GKE users

If there is no GKE k8s cluster available, there is still an option of trying out this example on your laptop or server.
Please refer to [MiniKF Readme](https://github.com/ciscoAI/KFLab/blob/master/tf-mnist/minikf)

# Prerequisites

## Google Kubernetes Engine Prerequisites

### Google Cloud Account Users

1. Create a [Google Cloud Account](https://console.cloud.google.com/)

2. Navigate in the Google Cloud Console to Google Kubernetes Engine and create a cluster.

3. Click on the 'Create Cluster' option to create a cluster with 3 `n1-standard-2` nodes and kubernetes master version as `1.11.7-gke.12` or above.

4. **gcloud**

    [Install gcloud SDK](https://cloud.google.com/deployment-manager/docs/step-by-step-guide/installation-and-setup) by skipping to the 'Install Cloud SDK' section given in the hyperlink.

5. Execute `gcloud auth login` command on your shell to authenticate your gcloud account.

6. Press the 'connect' button against your created cluster, then copy the command it provides you and execute it on your shell.

7. `kubectl config current-context` should return your cluster name which you created.

8. Create a cluster role for your user by running:  
`kubectl create clusterrolebinding your-user-cluster-admin-binding --clusterrole=cluster-admin --user=<your@email.com>`

### Service Account Users

1. Each team/person will be given a service account.
2. Install [gcloud](https://cloud.google.com/sdk/docs/quickstart-macos) on your
   macbooks.
3. Install [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/) on
   your macbooks.
4. Activate service account (```service-acc-name``` and ```json-file-name``` will be
   provided)
```
gcloud auth activate-service-account <service-acc-name> --key-file=<json-file-name> --project=ml-bootcamp-2018

Example (uses team-blr-1):
gcloud auth activate-service-account team-blr-1@ml-bootcamp-2018.iam.gserviceaccount.com --key-file=ml-bootcamp-2018-409e6b4a7257.json --project=ml-bootcamp-2018

```
5. Get kubeconfig for your cluster (```cluster-name``` and ```zone-name``` will
   be provided)
```
gcloud container clusters get-credentials <cluster-name> --zone <zone-name>

Example (uses team-blr-1):
gcloud container clusters get-credentials team-blr-1 --zone asia-south1-c
```
6. Enable admin cluster role binding (```your-user-cluster-admin-binding``` was
   retrieved in the previous step) (only 1 team member should run the below command)
```
kubectl create clusterrolebinding your-user-cluster-admin-binding --clusterrole=cluster-admin --user=<service-acc-name>

Example (uses team-blr-1):
kubectl create clusterrolebinding your-user-cluster-admin-binding --clusterrole=cluster-admin --user=team-blr-1@ml-bootcamp-2018.iam.gserviceaccount.com
```

## Setup CLI tools and other accounts

1. **kubectl cli**

   Install (kubectl)[https://kubernetes.io/docs/tasks/tools/install-kubectl/] if not present already.  
   Check if kubectl  is configured properly by accessing the cluster Info of your kubernetes cluster

        $ kubectl cluster-info

2. **ksonnet**

    Install [ksonnet](https://ksonnet.io/get-started/).  
    Check ksonnet version

        $ ks version

    Ksonnet version must be greater than or equal to **0.13.1**. Upgrade to the latest if it is an older version

3. Create a [github account](https://github.com) to be able to clone this repository.

If above commands succeeds, you are good to go !

# Installation

        ./install.bash


        # Ensure that all pods are running in the namespace set in variables.bash. The default namespace is kubeflow
        kubectl get pods -n kubeflow

If there is any rate limit error from github, please follow the instructions at:
[Github Token Setup](https://github.com/ksonnet/ksonnet/blob/master/docs/troubleshooting.md#github-rate-limiting-errors)


# Setup

1.  (**Optional**) If you want to use a custom image for training, create the training Image and upload to DockerHub. Else, skip this step to use the already existing image (`gcr.io/cpsg-ai-demo/tf-mnist-demo:v1`).

   Point `DOCKER_BASE_URL` to your DockerHub account. Point `IMAGE` to your training image. If you don't have a DockerHub account,
   create one at [https://hub.docker.com/](https://hub.docker.com/), upload your image there, and do the following
   (replace <username> and <container> with appropriate values).

       ```
       DOCKER_BASE_URL=docker.io/<username>
       IMAGE=${DOCKER_BASE_URL}/<image>
       docker build . --no-cache  -f Dockerfile -t ${IMAGE}
       docker push ${IMAGE}
       ```

> **NOTE.** Images kept in gcr.io might make things faster since it keeps images within GKE, thus avoiding delays of accessing the image
> from a remote container registry.


2. Run the training job setup script

       ```
	   ./train.bash
       # Ensure that all pods are running in the namespace set in variables.bash. The default namespace is kubeflow
       kubectl get pods -n kubeflow
       ```

3. Start TF serving on the trained results

       ```
       ./serve.bash
       ```

# Model Testing

The model can be tested using a local python client or via web application

## Using a local python client

This is the easiest way to test your model if your kubernetes cluster does not
support external loadbalancers. It uses port forwarding to expose the serving
service for the local clients.

Port forward to access the serving port locally

    ./portf.bash


Run a sample client code to predict images(See mnist-client.py)

    virtualenv --system-site-packages env
    source ./env/bin/activate
    easy_install -U pip
    pip install --upgrade tensorflow
    pip install tensorflow-serving-api
    pip install python-mnist
    pip install Pillow

    TF_MNIST_IMAGE_PATH=data/7.png python mnist_client.py

You should see the following result

    Your model says the above number is... 7!

Now try a different image in `data` directory :)

## Using a web application
### LoadBalancer

This is ideal if you would like to create a test web application exposed by a loadbalancer.

    MNIST_SERVING_IP=`kubectl -n ${NAMESPACE} get svc/mnist --output=jsonpath={.spec.clusterIP}`
    echo "MNIST_SERVING_IP is ${MNIST_SERVING_IP}"

Create image using Dockerfile in the webapp folder and upload to DockerHub

    CLIENT_IMAGE=${DOCKER_BASE_URL}/mnist-client
    docker build . --no-cache  -f Dockerfile -t ${CLIENT_IMAGE}
    docker push ${CLIENT_IMAGE}

    echo "CLIENT_IMAGE is ${CLIENT_IMAGE}"
    ks generate tf-mnist-client tf-mnist-client --mnist_serving_ip=${MNIST_SERVING_IP} --image=${CLIENT_IMAGE}

    ks apply ${KF_ENV} -c tf-mnist-client

    #Ensure that all pods are running in the namespace set in variables.bash.
    kubectl get pods -n ${NAMESPACE}

Now get the loadbalancer IP of the tf-mnist-client service

    kubectl get svc/tf-mnist-client -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

Open browser and see app at http://LoadBalancerIP

### Port Forwarding

A simple way to expose your web application is by port forwarding the mnist client service to your laptop. 

 ```
   ./webapp.bash
 ```

After running this script, open browser and see app at http://127.0.0.1:9001

### NodePort

Another way to expose your web application on the Internet is NodePort. Define
variables in variables.bash and run the following script:

 ```
   ./webapp.bash
 ```

After running this script, you will get the IP adress of your web application.
Open browser and see app at http://IP_ADRESS:NodePort

# Extras

## Using Persistent Volumes

Currently, this example uses a [NFS
server](https://github.com/CiscoAI/kubeflow-workflows/blob/d6d002f674c2201ec449ebd1e1d28fb335a64d1e/mnist/install.bash#L53)
that is automatically deployed in your Kubernetes cluster. If you have an
external NFS server, you can provide its IP during the nfs-volume deployment.
See [nfs-volume
deployment](https://github.com/CiscoAI/kubeflow-workflows/blob/d6d002f674c2201ec449ebd1e1d28fb335a64d1e/mnist/install.bash#L61)
step.

If you would like to have a different persistent volume, you can create a
Kubernetes
[pvc](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)  with the
value set in the variables.bash file. See
[`NFS_PVC_NAME`](https://github.com/CiscoAI/kubeflow-workflows/blob/d6d002f674c2201ec449ebd1e1d28fb335a64d1e/mnist/variables.bash#L19)
variable


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
## Clean up pods

	./cleanup.bash

   Forcefully terminate pods using:

   	$ kubectl delete pod <pod_name> --force -n kubeflow --grace-period=0

### Note

If container needs to use an HTTP, HTTPS, or FTP proxy server (for internet connectivity), configure it by setting the environment variables when building docker image. Set the HTTP, HTTPS, or FTP proxy server environment variable in Dockerfile.

	ENV HTTPS_PROXY "https://127.0.0.1:3001"
	ENV HTTP_PROXY "http://127.0.0.1:3001"
	ENV FTP_PROXY "ftp://127.0.0.1:3001"
	ENV NO_PROXY "*.test.example.com,.example2.com"
