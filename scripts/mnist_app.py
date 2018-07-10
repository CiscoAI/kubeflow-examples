import argparse
import os
import datetime
import logging
import subprocess
import time
import sys
#from termcolor import colored

import random
import numpy
from PIL import Image

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from grpc.beta import implementations

from mnist import MNIST # pylint: disable=no-name-in-module
from random import randint

from kubernetes import client, config
import util

MASTER_REPO_OWNER = "CiscoAI"
MASTER_REPO_NAME = "kubeflow-workflows"
port_forward_pid = 0

def clone_repo(dest,
               repo_owner=MASTER_REPO_OWNER,
               repo_name=MASTER_REPO_NAME,
               sha=None,
               branches=None):
  """Clone the repo,

  Args:
    dest: This is the root path for the training code.
    repo_owner: The owner for github organization.
    repo_name: The repo name.
    sha: The sha number of the repo.
    branches: (Optional): One or more branches to fetch. Each branch be specified
      as "remote:local". If no sha is provided
      we will checkout the last branch provided. If a sha is provided we
      checkout the provided sha.

  Returns:
    dest: Directory where it was checked out
    sha: The sha of the code.
  """
  # Clone CiscoAI
  repo = "https://github.com/{0}/{1}.git".format(repo_owner, repo_name)
  logging.info("repo %s", repo)

  util.run(["git", "clone", repo, dest])

  if branches:
    for b in branches:
      util.run(
        [
          "git",
          "fetch",
          "origin",
          b,
        ], cwd=dest)

    if not sha:
      b = branches[-1].split(":", 1)[-1]
      util.run(
        [
          "git",
          "checkout",
          b,
        ], cwd=dest)

  if sha:
    util.run(["git", "checkout", sha], cwd=dest)

  # Get the actual git hash.
  # This ensures even for periodic jobs which don't set the sha we know
  # the version of the code tested.
  #sha = util.run(["git", "rev-parse", "HEAD"], cwd=dest)

  return dest


'''def create_gcloud_cluster(project, zone):
  cmd = "gcloud config set project " + project
  util.run(cmd.split())
  cmd = "gcloud config set account nightlycpsg@cpsg-ai-kubeflow.iam.gserviceaccount.com"
  util.run(cmd.split())
  cmd = "gcloud auth activate-service-account --key-file=" + os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
  util.run(cmd.split())
  cmd = "gcloud container clusters create nightly --zone "+zone+" --num-nodes=5 --machine-type n1-standard-2"
  util.run(cmd.split())
  cmd = "gcloud config set container/cluster nightly"
  util.run(cmd.split())
  cmd = "gcloud container clusters get-credentials nightly --zone "+zone
  util.run(cmd.split())
  time.sleep(300)
  cmd = "kubectl create clusterrolebinding cluster-admin-binding --clusterrole=cluster-admin --user=nightlycpsg@cpsg-ai-kubeflow.iam.gserviceaccount.com"
  util.run(cmd.split())

def delete_gcloud_cluster(zone):
  cmd = "gcloud container clusters delete nightly --zone "+zone+" --quiet"
  util.run(cmd.split())'''

def check_data_export(project):
  cmd = "gcloud compute --project "+project+" ssh singlefs-2-vm --zone asia-southeast1-a --command"
  cmd1 = cmd.split()
  cmd1.append("ls /data/export")
  util.run(cmd1)

def rm_data_export(project):
  cmd = "gcloud compute --project "+project+" ssh singlefs-2-vm --zone asia-southeast1-a --command"
  cmd1 = cmd.split()
  cmd1.append("sudo rm -rf /data/export")
  util.run(cmd1)

def get_cluster_info():
  cmd = "kubectl cluster-info"
  util.run(cmd.split())

def get_pods():
  cmd = "kubectl get pods -n kubeflow"
  util.run(cmd.split())

def mnist_client():

  num_img = randint(0, 9)
  TF_MODEL_SERVER_HOST = os.getenv("TF_MODEL_SERVER_HOST", "127.0.0.1")
  TF_MODEL_SERVER_PORT = int(os.getenv("TF_MODEL_SERVER_PORT", 9000))
  TF_DATA_DIR = os.getenv("TF_DATA_DIR", "/tmp/data/")
  TF_MNIST_IMAGE_PATH = os.getenv("TF_MNIST_IMAGE_PATH", "data/"+str(num_img)+".png")
  TF_MNIST_TEST_IMAGE_NUMBER = int(os.getenv("TF_MNIST_TEST_IMAGE_NUMBER", -1))

  if TF_MNIST_IMAGE_PATH != None:
    raw_image = Image.open(TF_MNIST_IMAGE_PATH)
    int_image = numpy.array(raw_image)
    image = numpy.reshape(int_image, 784).astype(numpy.float32)
  elif TF_MNIST_TEST_IMAGE_NUMBER > -1:
    test_data_set = input_data.read_data_sets(TF_DATA_DIR, one_hot=True).test
    image = test_data_set.images[TF_MNIST_TEST_IMAGE_NUMBER]
  else:
    test_data_set = input_data.read_data_sets(TF_DATA_DIR, one_hot=True).test
    image = random.choice(test_data_set.images)

  channel = implementations.insecure_channel(
      TF_MODEL_SERVER_HOST, TF_MODEL_SERVER_PORT)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = "mnist"
  request.model_spec.signature_name = "serving_default"
  request.inputs['x'].CopyFrom(
      tf.contrib.util.make_tensor_proto(image, shape=[1, 28, 28]))

  result = stub.Predict(request, 10.0)  # 10 secs timeout

  logging.info(MNIST.display(image, threshold=0))
  logging.info("Your model says the above number is... %d!" %
      result.outputs["classes"].int_val[0])
  if (num_img == result.outputs["classes"].int_val[0]):
    #logging.info(colored("TEST PASSED!!!", 'green'))
    logging.info("TEST PASSED!!!")

def port_forward(app):

  config.load_kube_config()

  v1=client.CoreV1Api()
  print("Listing pods with their IPs:")
  ret = v1.list_namespaced_pod("kubeflow",watch=False)
  for i in ret.items:
    if((i.metadata.labels.get("app") != None) & (i.metadata.labels.get("app") == app)):
      serving_pod_name = i.metadata.name

  cmd = "kubectl -n kubeflow port-forward " + serving_pod_name + " 9000:9000"
  subprocess.Popen(cmd.split())

def check_train_job(app):

  config.load_kube_config()

  cnt = 0
  tjob = "tf-"+app+"job"

  v1=client.CoreV1Api()
  ret = v1.list_namespaced_pod("kubeflow",watch=False)
  for i in ret.items:
    if((i.metadata.labels.get("tf_job_name") != None) & (i.metadata.labels.get("tf_job_name") == tjob)):
      cnt = cnt + 1
  if(cnt == 3):
    logging.info("job instances spawned")
  else:
    logging.info("job instances not spawned")
    return 0

  time.sleep(300)

  cnt = 0
  ret = v1.list_namespaced_pod("kubeflow",watch=False)
  for i in ret.items:
    if((i.metadata.labels.get("tf_job_name") != None) & (i.metadata.labels.get("tf_job_name") == tjob)):
      cnt = cnt + 1
  if(cnt == 0):
    logging.info("job instances terminated ")
    return 1
  else:
    logging.info("job instances NOT terminated ")
    return 0

def check_serve_status(app):

  config.load_kube_config()

  v1=client.CoreV1Api()
  print("Listing pods with their IPs:")
  ret = v1.list_namespaced_pod("kubeflow",watch=False)
  for i in ret.items:
    if((i.metadata.labels.get("app") != None) & (i.metadata.labels.get("app") == app)):
      if(i.status.phase == "Running"):
        logging.info("Ready to serve")
        return 1
      else:
        logging.info("NOT Ready")
        return 0

def port_forward_start():

  devnull = open(os.devnull, 'wb') # Use this in Python < 3.3
  subprocess.Popen(['nohup', './portf.bash'], stdout=devnull, stderr=devnull)
  time.sleep(5)
  p = subprocess.Popen(['ps', '-aux'], stdout=subprocess.PIPE)
  out, err = p.communicate()
  for line in out.splitlines():
    if 'port-forward' in line:
      port_forward_pid = int(line.split()[1])
      logging.info(port_forward_pid)

def port_forward_stop():
  #p = subprocess.Popen(['ps', '-aux'], stdout=subprocess.PIPE)
  #out, err = p.communicate()
  #for line in out.splitlines():
  #  if 'port-forward' in line:
  #    pid = int(line.split()[1])
  print(port_forward_pid)
  #os.kill(port_forward_pid, 9)


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
    default="cpsg-ai-kubeflow",
    type=str,
    help="The project containing the GKE cluster.")

  parser.add_argument(
    "--zone",
    default="us-west1-b",
    type=str,
    help="The zone to create the GKE cluster.")

  parser.add_argument(
    "--name",
    default="mnist",
    type=str,
    help="The zone to create the GKE cluster.")

  parser.add_argument(
    "--app",
    default="mnist",
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
  test_log = os.path.join(args.logpath, "test-"+timestamp+".log")
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
  util.create_gcloud_cluster(args.project, args.zone, args.name)
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

  time.sleep(90)
  get_pods()
  util.run(["./train.bash"])
  ret = check_train_job(app)
  if not ret:
    util.run(["./cleanup.bash"])
    time.sleep(60)
    util.delete_gcloud_cluster(args.zone, args.name)
    sys.exit(1)
  #check_data_export(args.project)
  util.run(["./serve.bash"])
  time.sleep(60)
  ret = check_serve_status(args.app)
  if not ret:
    util.run(["./cleanup.bash"])
    time.sleep(60)
    util.delete_gcloud_cluster(args.zone, args.name)
    sys.exit(1)
  port_forward_start()
  time.sleep(5)
  mnist_client()
  time.sleep(5)
  util.run(["./cleanup.bash"])
  time.sleep(60)
  #rm_data_export(args.project)
  util.run(["rm","-rf","mnist"])
  util.delete_gcloud_cluster(args.zone, args.name)
  #os.chdir("../../")
  #file_handler.flush()
  #util.run(["gsutil","cp",test_log,"gs://cpsg-ai-kubeflow-bucket/nightly_logs/"])
  sys.exit(0)
