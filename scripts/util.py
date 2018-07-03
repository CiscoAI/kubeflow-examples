import os
import datetime
import time
import logging
import subprocess

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

def create_gcloud_cluster(project, zone):
  cmd = "gcloud config set project " + project
  run(cmd.split())
  cmd = "gcloud config set account nightlycpsg@cpsg-ai-kubeflow.iam.gserviceaccount.com"
  run(cmd.split())
  cmd = "gcloud auth activate-service-account --key-file=" + os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
  run(cmd.split())
  cmd = "gcloud container clusters create nightly --zone "+zone+" --num-nodes=5 --machine-type n1-standard-2"
  run(cmd.split())
  cmd = "gcloud config set container/cluster nightly"
  run(cmd.split())
  cmd = "gcloud container clusters get-credentials nightly --zone "+zone
  run(cmd.split())
  time.sleep(300)
  cmd = "kubectl create clusterrolebinding cluster-admin-binding --clusterrole=cluster-admin --user=nightlycpsg@cpsg-ai-kubeflow.iam.gserviceaccount.com"
  run(cmd.split())

def delete_gcloud_cluster(zone):
  cmd = "gcloud container clusters delete nightly --zone "+zone+" --quiet"
  run(cmd.split())
