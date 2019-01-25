#!/usr/local/env bash

## Namespace to be used in k8s cluster for your application
NAMESPACE=default

## Ksonnet app name
APP_NAME=kubebench

## GITHUB version for official kubeflow components
KUBEFLOW_GITHUB_VERSION=v0.3.0-rc.3

## GITHUB version for ciscoai components
CISCOAI_GITHUB_VERSION=master

## GITHUB version for kubebench components
KB_VERSION=v0.3.0
## Ksonnet environment name
KF_ENV=nativek8s

## Name of the NFS Persistent Volume
CONFIG_NAME="job-config"
JOB_NAME="my-benchmark"
CONFIG_PVC_NAME="kubebench-config-pvc"
DATA_PVC_NAME="kubebench-data-pvc"
EXP_PVC_NAME="kubebench-exp-pvc"
CONFIG_PVC_MOUNT="/exports/kubebench/config"
DATA_PVC_MOUNT="/exports/kubebench/data"
EXP_PVC_MOUNT="/exports/kubebench/experiments"

