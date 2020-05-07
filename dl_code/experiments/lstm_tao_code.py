#!/usr/bin/env python3

import os

from jobmonitor.api import (
    kubernetes_schedule_job,
    kubernetes_schedule_job_queue,
    register_job,
    upload_code_package,
)
from jobmonitor.connections import mongo

excluded_files = [
    "core",
    "output.tmp",
    ".vscode",
    "node_modules",
    "scripts",
    "data",
    ".git",
    "*.pyc",
    "._*",
    "__pycache__",
    "*.pdf",
    "*.js",
    "*.yaml",
    ".pylintrc",
    ".gitignore",
    ".AppleDouble",
    ".jobignore",
]


project = "decentralized_powersgd"
experiment = os.path.splitext(os.path.basename(__file__))[0]
script = "main.py"
description = """
Let's see how we do on the LSTM
""".strip()
base_config = {
    "arch": "rnn_lm",
    "rnn_n_hidden": 650,
    "rnn_n_layers": 3,
    "rnn_bptt_len": 30,
    "rnn_clip": 0.4,
    "rnn_use_pretrained_emb": False,
    "rnn_tie_weights": True,
    "drop_rate": 0.4,
    "avg_model": True,
    "data": "wikitext2",
    "pin_memory": True,
    "batch_size": 32,
    "base_batch_size": 28.8,
    "num_workers": 1,
    "eval_freq": 4,
    "num_epochs": 200,
    "consensus_stepsize": 0.6,
    "partition_data": "random",
    "reshuffle_per_epoch": False,
    "stop_criteria": "epoch",
    "n_mpi_process": 32,
    "n_sub_process": 1,
    "world": "0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1",
    "on_cuda": True,
    "use_ipc": False,
    "comm_device": "cuda",
    "lr": 2.5,
    "lr_scaleup": True,
    "lr_scaleup_factor": "graph",
    "lr_warmup": True,
    "lr_warmup_epochs": 5,
    "lr_schedule_scheme": "custom_multistep",
    "lr_change_epochs": "150,225",
    "lr_decay": 10,
    "weight_decay": 0,
    "use_nesterov": False,
    "momentum_factor": 0,
    "graph_topology": "social",
    "track_time": False,
    "display_tracked_time": False,
    "evaluate_avg": True,
}
n_workers = 32


code_package, files_uploaded = upload_code_package(".", excludes=excluded_files + ["gossip_run.py"])
print("Uploaded {} files.".format(len(files_uploaded)))


def schedule(name, config, skip_existing=True):
    # Skip pre-existing entries
    if (
        skip_existing
        and mongo.job.count_documents({"project": project, "job": name, "experiment": experiment})
        > 0
    ):
        return
    config = {**base_config, **config}
    job_id = register_job(
        user="vogels",
        project=project,
        experiment=experiment,
        job=name,
        n_workers=n_workers,
        priority=10,
        config_overrides=config,
        runtime_environment={"clone": {"code_package": code_package}, "script": script},
        annotations={"description": description},
    )
    print(
        f"OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 mpirun --mca orte_base_help_aggregate 0 --mca btl_tcp_if_exclude docker0,lo --mca btl_smcuda_use_cuda_ipc 0 --prefix /home/lin/.openmpi --hostfile hostfile -x JOBMONITOR_RESULTS_DIR -x JOBMONITOR_METADATA_HOST -x JOBMONITOR_METADATA_PORT -x JOBMONITOR_METADATA_DB -x JOBMONITOR_METADATA_USER -x JOBMONITOR_METADATA_PASS -x JOBMONITOR_TIMESERIES_HOST -x JOBMONITOR_TIMESERIES_PORT -x JOBMONITOR_TIMESERIES_DB -x JOBMONITOR_TIMESERIES_USER -x JOBMONITOR_TIMESERIES_PASS -x DATA -n {n_workers} /home/lin/conda/envs/pytorch-py3.6/bin/jobrun --mpi {job_id}"
    )


# All-reduce baseline
for seed in [1]:
    schedule(
        f"dpsgd",
        dict(
            manual_seed=seed,
            optimizer="thijs-dpsgd",
            optimizer_rank=1,
            optimizer_warm_start=True,
            optimizer_num_iterations=1,
            base_batch_size=28.8,
        ),
        skip_existing=False,
    )
    schedule(
        f"parallel-choco-sign",
        dict(
            manual_seed=seed,
            optimizer="parallel_choco_v",
            base_batch_size=28.8,
            consensus_stepsize=0.6,
        ),
    )
    for i in [1, 2, 4, 8]:
        schedule(
            f"power-gossip-{i}",
            dict(
                manual_seed=seed,
                optimizer="thijs-power-gossip",
                optimizer_rank=1,
                optimizer_warm_start=True,
                optimizer_num_iterations=i,
                base_batch_size=28.8,
            ),
            skip_existing=False,
        )
