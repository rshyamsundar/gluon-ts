import itertools
import pandas as pd
from gluonts_sagemaker.results_iterator import (
    iter_training_jobs,
    parse_job_descr,
)
import numpy as np
from collections import namedtuple
from scipy import stats
import time
import functools
import re

from experiment_metadata import *


def get_cloudwatch_log(log_group, log_stream_name, boto_session):
    nft = "nextForwardToken"
    logs = boto_session.client("logs")
    getl = functools.partial(
        logs.get_log_events,
        logGroupName=log_group,
        logStreamName=log_stream_name,
        startFromHead=True,
    )
    response = getl()
    evs = [ev["message"] for ev in response["events"]]
    next_token = None
    while nft in response and next_token != response[nft]:
        next_token = response[nft]
        response = getl(nextToken=next_token)
        for ev in response["events"]:
            evs.append(ev["message"])
    return "\n".join(evs)


def get_log_stream_name_for_job(train_job_name, boto_session):
    logs = boto_session.client("logs")
    response = logs.describe_log_streams(
        logGroupName="/aws/sagemaker/TrainingJobs",
        logStreamNamePrefix=train_job_name,
    )
    stream_names = [strm["logStreamName"] for strm in response["logStreams"]]
    return stream_names[0]


def get_log_for_job(train_job_name):
    sn = get_log_stream_name_for_job(train_job_name, boto_session)
    return get_cloudwatch_log(
        log_group="/aws/sagemaker/TrainingJobs",
        log_stream_name=sn,
        boto_session=boto_session,
    )


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data[-3:])
    n = len(a)
    m, se = np.mean(a), stats.stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    if np.isnan(h):
        h = 0.0
    return f"{round(m, 4)}+/-{round(h, 4)}"


def mean_std(data):
    data = [x for x in data if not np.isnan(x)]
    if len(data) < 1:
        return "-"
    a = np.array(data[:3])
    m, std = np.mean(a), np.std(a)
    return f"{m:.3f}+/-{std:.3f}"
    # return f'{round(m, 2)}+/-{round(std, 2)}'


# DeepState         &  0.017+/-0.002 \\
# DeepState         &  0.010+/-0.000 \\
def parse_log(log):
    Results = namedtuple("Results", "crps")
    crps_scores = re.findall(
        ".*gluonts\[metric-mean_wQuantileLoss\]: ([-+]?(\d+(\.\d*)?|\.\d+))",
        log,
    )
    loss_scores = re.findall("'epoch_loss'=([-+]?(\d+(\.\d*)?|\.\d+))", log)

    # print(crps_scores)
    all_results = list()
    for crps in crps_scores:
        all_results.append(Results(float(crps[0])))
    return all_results


def get_best_scores(log, up_to_epoch=None):
    results = parse_log(log)
    results = results if up_to_epoch is None else results[:up_to_epoch]
    results.sort()
    return results[0]


def load_sagemaker_results(
    NameContains: str,
    max_jobs: int,
    CreationTimeAfter: str,
    CreationTimeBefore: str = None,
):
    rows = []
    jobs = iter_training_jobs(
        sm,
        NameContains=NameContains,
        StatusEquals="Completed",
        CreationTimeAfter=CreationTimeAfter,
        CreationTimeBefore=CreationTimeBefore,
        SortBy="CreationTime",
        SortOrder="Descending",
    )

    for job_descr in itertools.islice(jobs, max_jobs):
        print(job_descr["TrainingJobName"])
        # todo use paginator
        log = get_log_for_job(job_descr["TrainingJobName"])
        scores = get_best_scores(log, up_to_epoch=None)
        row = parse_job_descr(
            job_descr=job_descr,
            tags=sm.list_tags(ResourceArn=job_descr["TrainingJobArn"]),
        )
        # print(row)
        row["mean_wQuantileLoss"] = scores.crps
        # row['training_loss'] = scores.loss
        # avoid throttling errors
        rows.append(row)
        time.sleep(2.0)

    res_df = pd.DataFrame(rows)
    res_df.dataset = res_df.dataset.str.strip('"')

    return res_df


def pivot(res_df, metric: str = "CRPS", aggfunc=mean_std):
    renamed_df = (
        res_df[res_df.dataset.isin(datasets)]
        .rename(columns=metrics_rename)
        .replace({**sagemaker_program_rename})
    )
    pivot_df = renamed_df.pivot_table(
        values=metric,
        index=sagemaker_program_col,
        columns=dataset_col,
        aggfunc=aggfunc,
    )[datasets]
    pivot_df = pivot_df.reindex(index=sagemaker_program_rename.values())
    return pivot_df


if __name__ == "__main__":

    res_df = load_sagemaker_results(
        NameContains=NameContains,
        CreationTimeAfter=CreationTimeAfter,
        CreationTimeBefore=CreationTimeBefore,
        max_jobs=max_jobs,
    )
    print(res_df)

    for metric in metrics:
        print(f"Generating table for {metric}")
        pivot_df = pivot(res_df, metric)
        print(pivot_df.to_latex(na_rep="-", float_format="%.2f"))
