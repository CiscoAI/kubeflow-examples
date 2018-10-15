#!/usr/bin/env bash
# Uncomment the following two lines to step through each command and to print
# the command being executed.
#set -x
#trap read debug

# Start the training job

# read common variables (between installation, training, and serving)
source variables.bash

cd ${APP_NAME}
pwd

ks generate kubebench-job ${JOB_NAME} --name=${JOB_NAME}

ks param set ${JOB_NAME} name ${JOB_NAME}
ks param set ${JOB_NAME} namespace ${NAMESPACE}
ks param set ${JOB_NAME} experimentConfigPvc ${CONFIG_PVC_NAME}
ks param set ${JOB_NAME} experimentRecordPvc ${EXP_PVC_NAME}

ks apply ${KF_ENV} -c ${JOB_NAME}

# Check that the container is up and running
kubectl get pods -n ${NAMESPACE}
