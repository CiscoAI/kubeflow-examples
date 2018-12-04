#!/usr/bin/env bash
source variables.bash

cd ${APP_NAME}
pwd

ks delete ${KF_ENV} -c tfserving
kubectl get pods -n ${NAMESPACE}

JOB=tf-mnistjob
ks delete ${KF_ENV} -c ${JOB} 

ks delete ${KF_ENV} -c nfs-volume
kubectl get pv -n ${NAMESPACE}
kubectl get pvc -n ${NAMESPACE}

ks delete ${KF_ENV} -c nfs-server

ks delete ${KF_ENV} -c centraldashboard
ks delete ${KF_ENV} -c tf-job-operator
kubectl get pods -n ${NAMESPACE}

ks env rm ${KF_ENV}
kubectl delete namespace ${NAMESPACE} 
