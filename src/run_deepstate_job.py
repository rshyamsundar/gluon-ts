"""
This example shows how to run an arbitrary script in Sagemaker with Swist.
The sibling entry_point script is defined in generic_script.py.
If you want to just evaluate one model on a dataset, consider using gluonts_sagemaker/example_train.py directly
as shown in example/example_train.py.
"""

import isengard
import sagemaker

import gluonts
from gluonts_sagemaker.estimator import GluonTSFramework
import os
import boto3
from gluonts_sagemaker.train import get_metrics

region = "us-west-2"
account = "670864377759"
role = "admin"

client = isengard.Client()
# profile_name = "mlf-bench"
# todo pass the boto client to avoid defining an environment variable
# os.environ["AWS_PROFILE"] = profile_name

sess = client.get_boto3_session(account, role, region=region)
sm = sess.client("sagemaker")
sm_sess = sagemaker.Session(boto_session=sess)

role = "arn:aws:iam::670864377759:role/service-role/AmazonSageMaker-ExecutionRole-20181125T162939"
region_name = "us-west-2"

BUCKET = "mlf-bench-datasets"
HYPERPARAMS = "good"

DATASET = "neurips-datasets/unsplitted/non-rolling"

job_configs = [
    {
        "data-bucket": BUCKET,
        "data-prefix": f"{DATASET}/exchange_rate_nips",
        "num-test-dates": 1,
        "hyperparams": HYPERPARAMS,
        "dataset": "exchange-rate",
        "estimator": "DeepAR",
        "prediction-length": 150,
    },
]

num_runs = 1

for _ in range(num_runs):

    for config in job_configs:
        base_job_name = f'deepstate-Richard-paper-{config["dataset"]}'
        print(base_job_name)
        experiment = GluonTSFramework(
            entry_point="train_and_eval_deepstate.py",
            source_dir="/Users/rangapur/gluon-ts/src/gluonts/",
            dependencies=gluonts.__path__,
            hyperparameters=config,
            sagemaker_session=sm_sess,
            role=role,
            train_instance_type="ml.c5.9xlarge",
            train_instance_count=1,
            base_job_name=base_job_name,
            image_name=f"763104351884.dkr.ecr.{region_name}.amazonaws.com/mxnet-training:1.6.0-cpu-py36-ubuntu16.04",
        )

        # Sagemaker needs to get at least one input channel.
        # If you are using the dataset repository to download data instead of sagemaker
        # make sure to pass some valid input channel anyway (e.g. this)
        experiment.fit(
            inputs={"train": "s3://temp-us-west-2/dummy/"}, wait=False
        )

        job_name = experiment.latest_training_job.job_name
        print(job_name)
