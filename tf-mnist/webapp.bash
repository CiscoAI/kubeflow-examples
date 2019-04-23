#!/usr/bin/env bash

# read common variables (between installation, training, and serving)
source variables.bash

# define new variables
TF_MODEL_SERVER_HOST=`kubectl describe pod mnist -n ${NAMESPACE} | grep IP | sed -E 's/IP:[[:space:]]+//'`
CLIENT_IMAGE=${DOCKER_HUB}/${DOCKER_USERNAME}/${DOCKER_IMAGE}
MNIST_SERVING_IP=`kubectl -n ${NAMESPACE} get svc/mnist --output=jsonpath={.spec.clusterIP}`


# docker authorization
#if [ "${DOCKER_HUB}" = "docker.io" ]
#then
#    sudo docker login
#fi

# move to webapp folder
#cd ${WEBAPP_FOLDER}

# build an image passing correct IP and port
#sudo docker build . --no-cache  -f Dockerfile -t ${CLIENT_IMAGE}
#sudo docker push ${CLIENT_IMAGE}

# move to ksonnet project
#cd ../${APP_NAME}
cd ${APP_NAME}

# generate from local template to use NodePort
ks generate tf-mnist-client-local tf-mnist-client --mnist_serving_ip=${TF_MODEL_SERVER_HOST} --image=${CLIENT_IMAGE}

ks apply ${KF_ENV} -c tf-mnist-client

# ensure that all pods are running in the namespace set in variables.bash.
kubectl get pods -n ${NAMESPACE}

# get nodePort
#NODE_PORT=`kubectl get svc/tf-mnist-client -n ${NAMESPACE} -o jsonpath='{.spec.ports[0].nodePort}'`
#CLUSTER_IP=`hostname -I | awk '{print $1}'`
#echo "Visit your webapp at ${CLUSTER_IP}:${NODE_PORT}"

# sleep till the svc comes up
echo "Wait till the client svc comes up:"
sleep 30

#port forward mnist client port to a local port
kubectl -n ${NAMESPACE} port-forward svc/tf-mnist-client ${WEBAPP_PORT}:80 
