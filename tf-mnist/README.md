# Overview of the application
This tutorial contains instructions to build an **end to end kubeflow app** on a
Kubernetes cluster running on Google Kubernetes Engine (GKE) with minimal prerequisites.
*It should work on any other K8s cluster as well.*
The mnist model is trained and served from a NFS mount. This example is intended for
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

 2. **Ksonnet**

    Check ksonnet version

        $ ks version

    Ksonnet version must be greater than or equal to **0.11.0**. Upgrade to the latest if it is an older version

If above commands succeeds, you are good to go !


# Installation

        ./install.bash


        # Ensure that all pods are running in the namespace set in variables.bash. The default namespace is kubeflow
        kubectl get pods -n kubeflow

If there is any rate limit error from github, please follow the instructions at:
https://ksonnet.io/docs/tutorial#troubleshooting-github-rate-limiting-errors.


# Setup

1.  (**Optional**) If you want to use a custom image for training, create the training Image and upload to DockerHub. Else, skip this step to use the already existing image (`gcr.io/cpsg-ai-demo/tf-mnist-demo:v1`).

   Point `DOCKER_BASE_URL` to your DockerHub account. Point `IMAGE` to your training image. If you don't have a DockerHub account,
   create one at [https://hub.docker.com/](https://hub.docker.com/), upload your image there, and do the following
   (replace <username> and <container> with appropriate values).

       DOCKER_BASE_URL=docker.io/<username>
       IMAGE=${DOCKER_BASE_URL}/<image>
       docker build . --no-cache  -f Dockerfile -t ${IMAGE}
       docker push ${IMAGE}
       
> **NOTE.** Images kept in gcr.io might make things faster since it keeps images within GKE, thus avoiding delays of accessing the image
> from a remote container registry.


2. Run the training job setup script

       ./train.bash

       # Ensure that all pods are running in the namespace set in variables.bash. The default namespace is kubeflow
       kubectl get pods -n kubeflow

3. Start TF serving on the trained results

       ./serve.bash

# Model Testing

The model can be tested using a local python client or via web application

## Using a local python client

 This is the easiest way to test your model if your kubernetes cluster does not support external loadbalancers. It uses port forwarding to expose the serving service for the local clients.

 Port forward to access the serving port locally

      ./portf.bash


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

 This is ideal if you would like to create a test web application exposed by a loadbalancer. 

       MNIST_SERVING_IP=`kubectl -n ${NAMESPACE} get svc/mnist --output=jsonpath={.spec.clusterIP}`
       echo "MNIST_SERVING_IP is ${MNIST_SERVING_IP}"

 Create image using Dockerfile in the webapp folder and upload to DockerHub
     
      CLIENT_IMAGE=${DOCKER_BASE_URL}/mnist-client
      docker build . --no-cache  -f Dockerfile -t ${CLIENT_IMAGE}
      docker push ${CLIENT_IMAGE}

      echo "CLIENT_IMAGE is ${CLIENT_IMAGE}"
      ks generate tf-mnist-client tf-mnist-client --mnist_serving_ip=${MNIST_SERVING_IP} --image=${CLIENT_IMAGE}

      ks apply {KF_ENV} -c tf-mnist-client
      
      #Ensure that all pods are running in the namespace set in variables.bash. 
      kubectl get pods -n ${NAMESPACE}
 
  Now get the loadbalancer IP of the tf-mnist-client service
 
      kubectl get svc/tf-mnist-client -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

  Open browser and see app at http://LoadBalancerIP 
  
  
  # Extras
  
  ## Using Persistent Volumes
       
Currently, this example uses a [NFS server](https://github.com/CiscoAI/kubeflow-workflows/blob/d6d002f674c2201ec449ebd1e1d28fb335a64d1e/mnist/install.bash#L53) that is automatically deployed in your Kubernetes cluster. If you have an external NFS server, you can provide its IP during the nfs-volume deployment. See [nfs-volume deployment](https://github.com/CiscoAI/kubeflow-workflows/blob/d6d002f674c2201ec449ebd1e1d28fb335a64d1e/mnist/install.bash#L61) step.

If you would like to have a different persistent volume, you can create a Kubernetes [pvc](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)  with the value set in the variables.bash file. See [`NFS_PVC_NAME`](https://github.com/CiscoAI/kubeflow-workflows/blob/d6d002f674c2201ec449ebd1e1d28fb335a64d1e/mnist/variables.bash#L19) variable
       

## Retrain your model

     
If you want to change the training image, set `image` to your new training image. See the [prototype generation](https://github.com/CiscoAI/kubeflow-workflows/blob/d6d002f674c2201ec449ebd1e1d28fb335a64d1e/mnist/train.bash#L21)
 
        ks param set ${JOB} image ${IMAGE}
        
 If you would like to retrain the model(with a new image or not), you can delete the current training job and create a new one. See the [training](https://github.com/CiscoAI/kubeflow-workflows/blob/d6d002f674c2201ec449ebd1e1d28fb335a64d1e/mnist/train.bash#L28) step. 

         ks delete ${KF_ENV} -c ${JOB}  
         ks apply ${KF_ENV} -c ${JOB}
    
