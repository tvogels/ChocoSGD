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
Running Tao's code for ResNet20
""".strip()
base_config = {}
n_workers = 8


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
        f"OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 mpirun --mca orte_base_help_aggregate 0 --mca btl_tcp_if_exclude docker0,lo --mca btl_smcuda_use_cuda_ipc 0 -n {n_workers} jobrun --mpi {job_id}"
    )


# All-reduce baseline
for seed in [6, 5, 4]:
    schedule(f"choco-{seed}", dict(manual_seed=seed), skip_existing=False)
