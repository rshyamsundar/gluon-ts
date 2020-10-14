import argparse
from pathlib import Path

import boto3
import mxnet as mx

import numpy as np

from gluonts.dataset.common import load_datasets
from gluonts.mx.distribution.lds import ParameterBounds
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import (
    make_evaluation_predictions,
    backtest_metrics,
)
from gluonts.evaluation.backtest import serialize_message

import logging

from gluonts.mx.trainer import Trainer


logger = logging.getLogger(__name__)

LOCAL_DATASET_PATH = "/opt/ml/input/"


def download_from_s3(bucket: str, prefix: str, dest: str, boto_session=None):
    s3 = (
        boto_session.client("s3")
        if boto_session is not None
        else boto3.client("s3")
    )
    for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix)["Contents"]:
        key = obj["Key"]
        print(f'downloading object "{key}" from bucket "{bucket}" to "{dest}"')
        file_path = Path(dest + "/" + key)
        file_path.parents[0].mkdir(parents=True, exist_ok=True)
        print(bucket, key, str(file_path))
        s3.download_file(Bucket=bucket, Key=key, Filename=str(file_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-bucket", type=str, required=True)
    parser.add_argument("--data-prefix", type=str, required=True)
    parser.add_argument("--num-test-dates", type=int, required=True)
    parser.add_argument("--prediction-length", type=int, required=True)
    parser.add_argument(
        "--output-data-dir", type=str, default="/opt/ml/output/data/"
    )
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model/")
    parser.add_argument("--hyperparams", type=str, required=True)

    args, _ = parser.parse_known_args()

    Path(args.output_data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    download_from_s3(
        bucket=args.data_bucket,
        prefix=args.data_prefix,
        dest=LOCAL_DATASET_PATH,
    )

    metadata, train_ds, test_ds = load_datasets(
        metadata=Path(LOCAL_DATASET_PATH) / args.data_prefix / "metadata",
        train=Path(LOCAL_DATASET_PATH) / args.data_prefix / "train",
        test=Path(LOCAL_DATASET_PATH) / args.data_prefix / "test",
    )

    if args.hyperparams == "fast":
        trainer = Trainer(
            ctx=mx.context.cpu(),
            epochs=1,
            num_batches_per_epoch=1,
            hybridize=False,
        )
    else:
        trainer = Trainer(
            ctx=mx.context.cpu(),
            epochs=100,
            num_batches_per_epoch=50,
            hybridize=False,
        )

    estimator = DeepStateEstimator(
        freq=metadata.freq,
        prediction_length=metadata.prediction_length,
        cardinality=[len(train_ds)],
        trainer=trainer,
        noise_std_bounds=ParameterBounds(1e-3, 1.0),
    )
    agg_metrics, item_metrics = backtest_metrics(train_ds, test_ds, estimator,)

    print("CRPS:", agg_metrics["mean_wQuantileLoss"])
    for name, value in agg_metrics.items():
        print(f"metric-{name}", value)
        serialize_message(logger, f"metric-{name}", value)
