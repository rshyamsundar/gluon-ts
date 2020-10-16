from pathlib import Path
import tempfile

from gluonts_sagemaker.estimator import GluonTSFramework

from experiment_metadata import *


ROLLING = True


requirements_dot_txt_file_name = "requirements.txt"
requirements_dot_txt_file_content = (
    "git+https://github.com/awslabs/gluon-ts.git"
)

# only using temporary directory for demonstration
temp_dir = tempfile.TemporaryDirectory()
temp_dir_path = Path(temp_dir.name)

# create the requirements.txt file
with open(
    temp_dir_path / requirements_dot_txt_file_name, "w"
) as req_file:  # has to be called requirements.txt
    req_file.write(requirements_dot_txt_file_content)
my_requirements_txt_file_path = str(
    temp_dir_path / requirements_dot_txt_file_name
)
print(f"my_requirements_txt_file_path = '{my_requirements_txt_file_path}'")


rolling_str = "rolling" if ROLLING else "non-rolling"
for _ in range(num_runs):

    for config in get_job_configs(rolling=ROLLING):
        base_job_name = f'{NameContains}-{rolling_str}-{config["dataset"]}'
        print(base_job_name)
        experiment = GluonTSFramework(
            entry_point="train_and_eval_deepstate.py",
            source_dir="/Users/rangapur/gluon-ts/src/",
            dependencies=[my_requirements_txt_file_path],
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
