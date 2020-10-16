import isengard
import sagemaker

from collections import OrderedDict


region = "us-west-2"
account = "670864377759"
role = "admin"

client = isengard.Client()
# profile_name = "mlf-bench"
# todo pass the boto client to avoid defining an environment variable
# os.environ["AWS_PROFILE"] = profile_name

boto_session = client.get_boto3_session(account, role, region=region)
sm = boto_session.client("sagemaker")
sm_sess = sagemaker.Session(boto_session=boto_session)

role = "arn:aws:iam::670864377759:role/service-role/AmazonSageMaker-ExecutionRole-20181125T162939"
region_name = "us-west-2"

profile_name = "mlf-bench"

job_name_col = "sagemaker_job_name"
dataset_col = "dataset"
sagemaker_program_col = "sagemaker_program"

BUCKET = "mlf-bench-datasets"
HYPERPARAMS = "good"

DATASET = "neurips-datasets/unsplitted/"


NameContains = "Richard-deepstate-rolling"
CreationTimeAfter = "2020-10-14"
CreationTimeBefore = "2020-10-16"

datasets = [
    "exchange-rate",
    "solar",
    "electricity",
    "traffic",
    "wiki",
]

max_jobs = 15

metrics = ["CRPS"]
metrics_rename = OrderedDict({"mean_wQuantileLoss": "CRPS",})

sagemaker_program_rename = OrderedDict(
    {'"train_and_eval_deepstate.py"': "DeepState",}
)


num_runs = 1


def get_job_configs(rolling=False):
    non_rolling_folder = "non-rolling/" if not rolling else ""
    return [
        {
            "data-bucket": BUCKET,
            "data-prefix": f"{DATASET}{non_rolling_folder}electricity_nips",
            "hyperparams": HYPERPARAMS,
            "dataset": "electricity",
        },
        {
            "data-bucket": BUCKET,
            "data-prefix": f"{DATASET}{non_rolling_folder}exchange_rate_nips",
            "hyperparams": HYPERPARAMS,
            "dataset": "exchange-rate",
        },
        {
            "data-bucket": BUCKET,
            "data-prefix": f"{DATASET}{non_rolling_folder}solar_nips",
            "hyperparams": HYPERPARAMS,
            "dataset": "solar",
        },
        {
            "data-bucket": BUCKET,
            "data-prefix": f"{DATASET}{non_rolling_folder}traffic_nips",
            "hyperparams": HYPERPARAMS,
            "dataset": "traffic",
        },
        {
            "data-bucket": BUCKET,
            "data-prefix": f"{DATASET}{non_rolling_folder}wiki-rolling_nips",
            "hyperparams": HYPERPARAMS,
            "dataset": "wiki",
        },
    ]
