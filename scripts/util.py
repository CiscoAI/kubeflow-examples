import sys
import os
import datetime
import time
import logging
import subprocess
import argparse

def run(command,
        cwd=None,
        env=None,
        polling_interval=datetime.timedelta(seconds=1)):
  """Run a subprocess.

  Any subprocess output is emitted through the logging modules.

  Returns:
    output: A string containing the output.
  """
  logging.info("Running: %s \ncwd=%s", " ".join(command), cwd)

  if not env:
    env = os.environ
  else:
    keys = sorted(env.keys())

    lines = []
    for k in keys:
      lines.append("{0}={1}".format(k, env[k]))
    logging.info("Running: Environment:\n%s", "\n".join(lines))

  process = subprocess.Popen(
    command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

  logging.info("Subprocess output:\n")
  output = []
  while process.poll() is None:
    process.stdout.flush()
    for line in iter(process.stdout.readline, ''):
      output.append(line.strip())
      logging.info(line.strip())

    time.sleep(polling_interval.total_seconds())

  process.stdout.flush()
  for line in iter(process.stdout.readline, b''):
    output.append(line.strip())
    logging.info(line.strip())

  if process.returncode != 0:
    raise subprocess.CalledProcessError(
      process.returncode, "cmd: {0} exited with code {1}".format(
        " ".join(command), process.returncode), "\n".join(output))

  return "\n".join(output)

def create_gcloud_cluster(project, zone, name):
  cmd = "gcloud config set project " + project
  run(cmd.split())
  cmd = "gcloud config set account nightlycpsg@cpsg-ai-test.iam.gserviceaccount.com"
  run(cmd.split())
  cmd = "gcloud auth activate-service-account --key-file=" + os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
  run(cmd.split())
  cmd = "gcloud container clusters create " + name + " --zone "+zone+" --num-nodes=5 --machine-type n1-standard-2"
  run(cmd.split())
  cmd = "gcloud config set container/cluster " + name
  run(cmd.split())
  cmd = "gcloud container clusters get-credentials " + name + " --zone "+zone
  run(cmd.split())
  time.sleep(300)
  cmd = "kubectl create clusterrolebinding cluster-admin-binding --clusterrole=cluster-admin --user=nightlycpsg@cpsg-ai-test.iam.gserviceaccount.com"
  run(cmd.split())

def delete_gcloud_cluster(zone, name):
  cmd = "gcloud container clusters delete " + name + " --zone "+zone+" --quiet"
  run(cmd.split())

def get_cluster_info():
  cmd = "kubectl cluster-info"
  run(cmd.split())


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO,
                      format=('%(levelname)s|%(asctime)s'
                              '|%(pathname)s|%(lineno)d| %(message)s'),
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      )
  logging.getLogger().setLevel(logging.INFO)
  parser = argparse.ArgumentParser(
    description="Submit test to run the nightly .")

  parser.add_argument(
    "--project",
    default="cpsg-ai-test",
    type=str,
    help="The project containing the GKE cluster.")

  parser.add_argument(
    "--zone",
    default="asia-south1-a",
    type=str,
    help="The zone to create the GKE cluster.")

  parser.add_argument(
    "--name",
    default="test",
    type=str,
    help="The zone to create the GKE cluster.")

  parser.add_argument(
    "--create",
    default="no",
    type=str,
    help="create a GKE cluster.")

  parser.add_argument(
    "--delete",
    default="no",
    type=str,
    help="create a GKE cluster.")

  args = parser.parse_args()

  if(args.create == "yes"):
      create_gcloud_cluster(args.project, args.zone,args.name)

  get_cluster_info()
  if(args.delete == "yes"):
      delete_gcloud_cluster(args.zone,args.name)

  sys.exit(0)

