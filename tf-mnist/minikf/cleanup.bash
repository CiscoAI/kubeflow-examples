#!/usr/bin/env bash
source variables.bash

cd ${VAGRANT_KS_APP}
pwd

ks delete ${KF_ENV} -c tfserving
kubectl get pods -n ${NAMESPACE}

JOB=tf-${APP_NAME}job
ks delete ${KF_ENV} -c ${JOB} 

ks delete ${KF_ENV} -c nfs-volume
kubectl get pv -n ${NAMESPACE}
kubectl get pvc -n ${NAMESPACE}

ks delete ${KF_ENV} -c nfs-server
