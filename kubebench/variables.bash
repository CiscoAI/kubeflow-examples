#!/usr/local/env bash

## Namespace to be used in k8s cluster for your application
NAMESPACE=default

## Ksonnet app name
APP_NAME=kubebench

## GITHUB version for official kubeflow components
KUBEFLOW_GITHUB_VERSION=master

## GITHUB version for ciscoai components
CISCOAI_GITHUB_VERSION=master

## GITHUB version for kubebench components
KB_VERSION=master
## Ksonnet environment name
KF_ENV=nativek8s

## Name of the NFS Persistent Volume
CONFIG_NAME="job-config"
JOB_NAME="my-benchmark"
PVC_NAME="kubebench-pvc"
PVC_MOUNT="/kubebench"

