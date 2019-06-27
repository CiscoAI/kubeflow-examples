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
ks pkg install kubeflow/common@${KUBEFLOW_GITHUB_VERSION}
ks pkg install kubeflow/tf-training@${KUBEFLOW_GITHUB_VERSION}
ks pkg install kubeflow/tf-serving@${KUBEFLOW_GITHUB_VERSION}

ks pkg install ciscoai/tf-${APP_NAME}job@${CISCOAI_GITHUB_VERSION}

#6. Deploy kubeflow core components to K8s cluster.

# If you are doing this on GCP, you need to run the following command first:
# kubectl create clusterrolebinding your-user-cluster-admin-binding --clusterrole=cluster-admin --user=<your@email.com>

ks generate tf-job-operator tf-job-operator
ks apply ${KF_ENV} -c tf-job-operator

#### Installation is complete now ####

echo "Make sure that the pods are running"
kubectl get pods -n ${NAMESPACE}
