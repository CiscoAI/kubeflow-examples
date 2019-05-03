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

# Wait till the svc comes up
timeout="1000"
echo "Obtaining the pod name..."
start_time=`date +%s`
pod_name=""

while [[ $pod_name == "" ]];do
  pod_name=$(kubectl get pods --namespace "${NAMESPACE}" --selector=app=mnist-client --template '{{range .items}}{{.metadata.name}}{{"\n"}}{{end}}')
  current_time=`date +%s`
  elapsed_time=$(expr $current_time + 1 - $start_time)
  if [[ $elapsed_time -gt $timeout ]];then
    echo "timeout"
    exit 1
  fi
  sleep 2
done
echo "Pod name is: " $pod_name

# Wait for the pod container to start running
echo "Waiting for the TF Serving pod to start running..."
start_time=`date +%s`
exit_code="1"
while [[ $exit_code != "0" ]];do
  kubectl get pod ${pod_name} --namespace "${NAMESPACE}" -o jsonpath='{.status.containerStatuses[0].state.running}'
  exit_code=$?
  current_time=`date +%s`
  elapsed_time=$(expr $current_time + 1 - $start_time)
  if [[ $elapsed_time -gt $timeout ]];then
    echo "timeout"
    exit 1
  fi
  sleep 2
done

echo "TF MNIST client service created."

if [ $1 == "portforward" ]
then
#port-forward mnist client port to a local port
    kubectl -n ${NAMESPACE} port-forward svc/tf-mnist-client ${WEBAPP_PORT}:80

elif [ $1 == "nodeport" ]
then
#Prints Pod external IP  with Nodeport 
    POD_NAME=$(kubectl get pods --namespace ${NAMESPACE}  --selector=app=mnist-client  --template '{{range .items}}{{.metadata.name}}{{"\n"}}{{end}}')
    NODE_NAME=`kubectl get pod/$POD_NAME -n kubeflow -o jsonpath='{.spec.nodeName}'`
    NODE_IP=`kubectl get node/$NODE_NAME  -o jsonpath='{.status.addresses[1].address}'`
    NODE_PORT=`kubectl get svc/tf-mnist-client -n ${NAMESPACE} -o jsonpath='{.spec.ports[0].nodePort}'`
    echo "Webapp is running at http://${NODE_IP}:${NODE_PORT}"
fi
