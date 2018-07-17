# Overview of the application
The goal of Kubebench is to make it easy to run benchmark jobs on Kubeflow with various system and model settings. 
Kubebench enables benchmarks by leveraging Kubeflow's capability of managing TFJobs, as well as Argo based workflows.
The following demonstrates how to setup a kubebench end-to-end application.

# Prerequisites

1. **kubectl cli**

   Check if kubectl  is configured properly by accessing the cluster Info of your kubernetes cluster

       $kubectl cluster-info

 2. **Ksonnet**(Latest version:version: 0.11.0)

    Check ksonnet version

        $ ks version


If above commands succeeds, you are good to go !


# Installation

        ./install.bash


      # Ensure that all pods are running in the namespace set in variables.bash. 
      kubectl get pods -n default

If there is any rate limit error from github, please follow the instructions at:
https://ksonnet.io/docs/tutorial#troubleshooting-github-rate-limiting-errors.

# Setup
Run the kubebench job setup script

       ./run_kb.bash
    
       # Ensure that all pods are running in the namespace set in variables.bash. 
       kubectl get pods -n default
       
# Cleanup
Clean up the kubebench job, argo workflows, kubeflow 

       ./cleanup.bash

For more details, please refer https://github.com/kubeflow/kubebench/blob/master/README.md
