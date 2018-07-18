#!/usr/bin/env bash

# Uncomment the following two lines to step through each command and to print
# the command being executed.
#set -x
#trap read debug

#1. Read variables
source variables.bash

#2. Create namespace if not present
kubectl create namespace ${NAMESPACE}

#3. Initialize the ksonnet app and create ksonnet environment. Environment makes it easy to manage app versions(Say dev, prod, test)
ks init ${APP_NAME}
cd ${APP_NAME}
ks env add ${KF_ENV}
ks env set ${KF_ENV} --namespace ${NAMESPACE}

#4. Add Ksonnet registries for adding prototypes. Prototypes are ksonnet templates

## Public registry that contains the official kubeflow components
ks registry add kubeflow github.com/kubeflow/kubeflow/tree/${KUBEFLOW_GITHUB_VERSION}/kubeflow

## Private registry that contains ${APP_NAME} example components
ks registry add ciscoai github.com/CiscoAI/kubeflow-examples/tree/${CISCOAI_GITHUB_VERSION}/tf-${APP_NAME}/pkg

#5. Install necessary packages from registries

ks pkg install kubeflow/core@${KUBEFLOW_GITHUB_VERSION}
ks pkg install kubeflow/tf-serving@${KUBEFLOW_GITHUB_VERSION}
ks pkg install kubeflow/tf-job@${KUBEFLOW_GITHUB_VERSION}

ks pkg install ciscoai/nfs-server@${CISCOAI_GITHUB_VERSION}
ks pkg install ciscoai/nfs-volume@${CISCOAI_GITHUB_VERSION}
ks pkg install ciscoai/tf-${APP_NAME}job@${CISCOAI_GITHUB_VERSION}

#6. Deploy kubeflow core components to K8s cluster.

# If you are doing this on GCP, you need to run the following command first:
# kubectl create clusterrolebinding your-user-cluster-admin-binding --clusterrole=cluster-admin --user=<your@email.com>

ks generate kubeflow-core kubeflow-core
ks param set kubeflow-core tfJobImage "gcr.io/kubeflow-images-public/tf_operator:v20180522-77375baf"
ks param set kubeflow-core tfJobVersion v1alpha1
ks apply ${KF_ENV} -c kubeflow-core

#7. Deploy NFS server in the k8s cluster **(Optional step)**

# If you have already setup a NFS server, you can skip this step and proceed to
# step 8. Set `NFS_SERVER_IP`to ip of your NFS server
ks generate nfs-server nfs-server
ks apply ${KF_ENV} -c nfs-server

#8. Deploy NFS PV/PVC in the k8s cluster **(Optional step)**

# If you have already created NFS PersistentVolume and PersistentVolumeClaim,
# you can skip this step and proceed to step 9.
NFS_SERVER_IP=`kubectl -n ${NAMESPACE} get svc/nfs-server  --output=jsonpath={.spec.clusterIP}`
echo "NFS Server IP: ${NFS_SERVER_IP}"
ks generate nfs-volume nfs-volume  --name=${NFS_PVC_NAME}  --nfs_server_ip=${NFS_SERVER_IP}
ks apply ${KF_ENV} -c nfs-volume

#### Installation is complete now ####

echo "Make sure that the pods are running"
kubectl get pods -n ${NAMESPACE}

echo "If you have created NFS Persistent Volume, ensure PVC is created and status is BOUND"
kubectl get pvc -n ${NAMESPACE}
