import argparse
import os
import datetime
import logging
import subprocess
import time
import sys
#from termcolor import colored

import random
from random import randint

from kubernetes import client, config
import util

MASTER_REPO_OWNER = "CiscoAI"
MASTER_REPO_NAME = "kubeflow-workflows"

# Configure service account to grant Argo more privileges
def config_svc_account():
  cmd = "kubectl -n default create rolebinding default-admin --clusterrole=cluster-admin --serviceaccount=default:default"
  util.run(cmd.split())

def get_cluster_info():
  cmd = "kubectl cluster-info"
  util.run(cmd.split())

def get_pods():
  cmd = "kubectl get pods -n default"
  util.run(cmd.split())

def copy_job_config():

  config.load_kube_config()

  v1=client.CoreV1Api()
  nfs_server_pod = None
  ret = v1.list_namespaced_pod("default",watch=False)
  for i in ret.items:
    if((i.metadata.labels.get("role") != None) & (i.metadata.labels.get("role") == "nfs-server")):
      nfs_server_pod = i.metadata.name
  if(nfs_server_pod == None):
    logging.info("nfs server pod NOT found")
    return 0

  cmd = "kubectl -n default exec " + nfs_server_pod + " -- mkdir -p /exports/kubebench/config"
  util.run(cmd.split())
  cmd = "kubectl -n default exec " + nfs_server_pod + " -- mkdir -p /exports/kubebench/data"
  util.run(cmd.split())
  cmd = "kubectl -n default exec " + nfs_server_pod + " -- mkdir -p /exports/kubebench/experiments"
  util.run(cmd.split())
  cmd = "kubectl cp ../config/tf_cnn_benchmarks/tf-cnn-dummy.yaml default/" + nfs_server_pod + ":/exports/kubebench/config/tf-cnn-dummy.yaml"
  util.run(cmd.split())

def check_kb_job(app):

  config.load_kube_config()

  crd_api=client.CustomObjectsApi()
  GROUP = "argoproj.io"
  VERSION = "v1alpha1"
  PLURAL = "workflows"
  res = crd_api.get_namespaced_custom_object(GROUP, VERSION, "default", PLURAL, "my-benchmark")

  if(res["status"]["phase"] == "Succeeded"):
    logging.info("Job Completed")
    return 1
  else:
    logging.info("Job NOT Completed")
    return 0


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO,
                      format=('%(levelname)s|%(asctime)s'
                              '|%(pathname)s|%(lineno)d| %(message)s'),
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      )
  logging.getLogger().setLevel(logging.INFO)
  # create the top-level parser
  parser = argparse.ArgumentParser(
    description="Submit test to run the nightly .")

  parser.add_argument(
    "--project",
    default="cpsg-ai-test",
    type=str,
    help="The project containing the GKE cluster.")

  parser.add_argument(
    "--zone",
    default="us-west1-b",
    type=str,
    help="The zone to create the GKE cluster.")

  parser.add_argument(
    "--name",
    default="kubebench",
    type=str,
    help="The zone to create the GKE cluster.")

  parser.add_argument(
    "--app",
    default="kubebench",
    type=str,
    help="The app to run <mnist>/<>")

  parser.add_argument(
    "--repo",
    default="n_repo",
    type=str,
    help="The app to run <mnist>/<>")

  parser.add_argument(
    "--logpath",
    default="nightly_logs",
    type=str,
    help="The app to run <mnist>/<>")

  # Parse the args
  args = parser.parse_args()

  # Setup a logging file handler. This way we can upload the log outputs
  # to gubernator.
  root_logger = logging.getLogger()

  timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
  test_log = os.path.join(args.logpath, args.app+"-test-"+timestamp+".log")
  if not os.path.exists(os.path.dirname(test_log)):
    try:
      os.makedirs(os.path.dirname(test_log))
    # Ignore OSError because sometimes another process
    # running in parallel creates this directory at the same time
    except OSError:
      pass


  file_handler = logging.FileHandler(test_log)
  root_logger.addHandler(file_handler)
  # We need to explicitly set the formatter because it will not pick up
  # the BasicConfig.
  formatter = logging.Formatter(fmt=("%(levelname)s|%(asctime)s"
                                     "|%(pathname)s|%(lineno)d| %(message)s"),
                                datefmt="%Y-%m-%dT%H:%M:%S")
  file_handler.setFormatter(formatter)
  logging.info("Logging to %s", test_log)

  final_result = util.run(["ks", "version"])
  if not final_result:
    # Exit with a non-zero exit code by to signal failure to prow.
    logging.error("One or more test steps failed exiting with non-zero exit "
                  "code.")
  util.set_kube_config(args.project, args.zone, args.name)
  get_cluster_info()
  logging.info(args)
  #repo_dir = clone_repo("nightly_repo")
  repo_dir = args.repo
  app = args.app
  os.chdir(repo_dir + "/" + app)
  util.run(["ls"])
  final_result = util.run(["./install.bash"])
  if not final_result:
    # Exit with a non-zero exit code by to signal failure to prow.
    logging.error("One or more test steps failed exiting with non-zero exit "
                  "code.")
    util.run(["./cleanup.bash"])
    sys.exit(1)

  time.sleep(60)
  get_pods()
  config_svc_account()
  copy_job_config()
  util.run(["./run_kb.bash"])
  time.sleep(240)
  ret = check_kb_job(app)
  if not ret:
    logging.error("One or more test steps failed exiting with non-zero exit "
                  "code.")
    util.run(["./cleanup.bash"])
    sys.exit(1)
  util.run(["./cleanup.bash"])
  sys.exit(0)
