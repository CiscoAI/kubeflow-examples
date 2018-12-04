#!/usr/bin/env bash

# Uncomment the following two lines to step through each command and to print
# the command being executed.
#set -x
#trap read debug

# Start TF serving on the trained results

# Note that `tfserving's modelPath` is set to `tfmnistjob's TF_EXPORT_DIR` so
# that tf serving pod automatically picks up the training results when training
# is completed.

# Read variables
source variables.bash

#Get the labels for the serving pod
kubectl -n ${NAMESPACE} get pod --show-labels
# Port forward to access the serving port locally
SERVING_POD_NAME=`kubectl -n ${NAMESPACE} get pod -l=app=mnist -o jsonpath='{.items[0].metadata.name}'`
kubectl -n ${NAMESPACE} port-forward ${SERVING_POD_NAME} ${PORT}:9000
