# -*- coding: utf-8 -*-
import datetime
import os
from argparse import Namespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.create_optimizer as create_optimizer
import pcode.create_scheduler as create_scheduler
import pcode.utils.checkpoint as checkpoint
import pcode.utils.error_handler as error_handler
import pcode.utils.logging as logging
import pcode.utils.op_paths as op_paths
import pcode.utils.stat_tracker as stat_tracker
import pcode.utils.topology as topology
from parameters import get_args
from pcode.utils.timer import Timer

"""
Run this as
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 mpirun --mca orte_base_help_aggregate 0 --mca btl_tcp_if_exclude docker0,lo --mca btl_smcuda_use_cuda_ipc 0 -n 8 python main.py
"""

config = dict(
    adam_beta_1=0.9,
    adam_beta_2=0.999,
    adam_eps=1e-08,
    arch="resnet20",
    avg_model=True,
    backend="mpi",
    base_batch_size=24,
    batch_size=128,
    checkpoint_index=None,
    clean_python=False,
    clip_grad=False,
    clip_grad_val=None,
    comm_algo=None,
    comm_device="cuda",
    comm_op="sign",
    compress_ratio=0.9,
    compress_warmup_epochs=0,
    compress_warmup_values="0.75,0.9375,0.984375,0.996,0.999",
    consensus_stepsize=0.45,
    data="cifar10",
    data_dir="./data/",
    densenet_bc_mode=False,
    densenet_compression=0.5,
    densenet_growth_rate=12,
    display_tracked_time=True,
    drop_rate=0.0,
    eval_freq=1,
    evaluate=False,
    evaluate_avg=True,
    evaluate_consensus=False,
    graph_topology="ring",
    is_biased=True,
    local_adam_memory_treatment=None,
    local_rank=None,
    local_step=1,
    lr=0.1,
    lr_alpha=None,
    lr_change_epochs="150,225",
    lr_decay=10,
    lr_fields=None,
    lr_gamma=None,
    lr_mu=None,
    lr_onecycle_extra_low=0.0015,
    lr_onecycle_high=3,
    lr_onecycle_low=0.15,
    lr_onecycle_num_epoch=46,
    lr_scale_indicators=None,
    lr_scaleup=True,
    lr_scaleup_factor="graph",
    lr_scaleup_type="linear",
    lr_schedule_scheme="custom_multistep",
    lr_warmup=True,
    lr_warmup_epochs=5,
    majority_vote=False,
    manual_seed=6,
    mask_momentum=False,
    momentum_factor=0.9,
    mpi_env=None,
    mpi_path="/usr/local",
    n_mpi_process=8,
    n_sub_process=1,
    num_epochs=300,
    num_iterations=32000,
    num_workers=0,
    on_cuda=True,
    optimizer="parallel_choco",
    partition_data="random",
    pin_memory=False,
    project="choco",
    python_path="/opt/anaconda3/bin/python",
    quantize_level=16,
    remote_exec=False,
    reshuffle_per_epoch=True,
    resume=None,
    rnn_bptt_len=35,
    rnn_clip=0.25,
    rnn_n_hidden=200,
    rnn_n_layers=2,
    rnn_tie_weights=True,
    rnn_use_pretrained_emb=True,
    rnn_weight_norm=False,
    save_all_models=False,
    save_some_models=None,
    stop_criteria="epoch",
    summary_freq=100,
    timestamp="1587804921_l2-0.0005_lr-0.01_epochs-90_batchsize-256_basebatchsize-None_num_mpi_process_1_n_sub_process-1_topology-complete_optim-sgd_comm_info-",
    track_detailed_time=False,
    track_time=True,
    train_fast=True,
    turn_on_local_step_from=0,
    use_ipc=False,
    use_lmdb_data=False,
    use_nesterov=True,
    user="lin",
    weight_decay=1e-4,
    wideresnet_widen_factor=4,
    work_dir=None,
    world="0,0,0,0,0,0,0,0",
)

output_dir = "output.tmp"


def main():
    # Convert config handling between Thijs and Tao's approach
    config["checkpoint"] = output_dir
    config["data_dir"] = os.path.join(os.getenv("DATA"), "data")
    conf = Namespace()
    for key, value in config.items():
        conf.__setattr__(key, value)

    if conf.optimizer == "parallel_choco":
        mp.set_start_method("forkserver", force=True)
        # mp.set_start_method("spawn", force=True)
        mp.set_sharing_strategy("file_system")

    try:
        init_distributed_world(conf, backend=conf.backend)
        conf.distributed = True and conf.n_mpi_process > 1
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False

    # init the config.
    init_config(conf)

    # define the timer for different operations.
    # if we choose the `train_fast` mode, then we will not track the time.
    conf.timer = Timer(
        verbosity_level=1 if conf.track_time and not conf.train_fast else 0, log_fn=log_metric
    )

    # create dataset.
    data_loader = create_dataset.define_dataset(conf, force_shuffle=True)

    # create model
    model = create_model.define_model(conf, data_loader=data_loader)

    # define the optimizer.
    optimizer = create_optimizer.define_optimizer(conf, model)

    # define the lr scheduler.
    scheduler = create_scheduler.Scheduler(conf)

    # add model with data-parallel wrapper.
    if conf.graph.on_cuda:
        if conf.n_sub_process > 1:
            model = torch.nn.DataParallel(model, device_ids=conf.graph.device)

    # (optional) reload checkpoint
    try:
        checkpoint.maybe_resume_from_checkpoint(conf, model, optimizer, scheduler)
    except RuntimeError as e:
        conf.logger.log(f"Resume Error: {e}")
        conf.resumed = False

    # train amd evaluate model.
    if "rnn_lm" in conf.arch:
        from pcode.distributed_running_nlp import train_and_validate

        # safety check.
        assert conf.n_sub_process == 1, "our current data-parallel wrapper does not support RNN."

        # define the criterion and metrics.
        criterion = nn.CrossEntropyLoss(reduction="mean")
        criterion = criterion.cuda() if conf.graph.on_cuda else criterion
        metrics = create_metrics.Metrics(
            model.module if "DataParallel" == model.__class__.__name__ else model,
            task="language_modeling",
        )

        # define the best_perf tracker, either empty or from the checkpoint.
        best_tracker = stat_tracker.BestPerf(
            best_perf=None if "best_perf" not in conf else conf.best_perf, larger_is_better=False
        )
        scheduler.set_best_tracker(best_tracker)

        # get train_and_validate_func
        train_and_validate_fn = train_and_validate
    else:
        from pcode.distributed_running_cv import train_and_validate

        # define the criterion and metrics.
        criterion = nn.CrossEntropyLoss(reduction="mean")
        criterion = criterion.cuda() if conf.graph.on_cuda else criterion
        metrics = create_metrics.Metrics(
            model.module if "DataParallel" == model.__class__.__name__ else model,
            task="classification",
        )

        # define the best_perf tracker, either empty or from the checkpoint.
        best_tracker = stat_tracker.BestPerf(
            best_perf=None if "best_perf" not in conf else conf.best_perf, larger_is_better=True
        )
        scheduler.set_best_tracker(best_tracker)

        # get train_and_validate_func
        train_and_validate_fn = train_and_validate

    # save arguments to disk.
    checkpoint.save_arguments(conf)

    # start training.
    train_and_validate_fn(
        conf,
        model=model,
        criterion=criterion,
        scheduler=scheduler,
        optimizer=optimizer,
        metrics=metrics,
        data_loader=data_loader,
        log_progress=log_progress,
    )

    for entry in conf.timer.transcript():
        log_runtime(entry["event"], entry["mean"], entry["std"], entry["instances"])

    # temporarily hack the exit parallelchoco
    if optimizer.__class__.__name__ == "ParallelCHOCO":
        error_handler.abort()


def log_progress(progress):
    info({"state.progress": progress})


def init_config(conf):
    # define the graph for the computation.
    cur_rank = dist.get_rank() if conf.distributed else 0
    conf.graph = topology.define_graph_topology(
        graph_topology=conf.graph_topology,
        world=conf.world,
        n_mpi_process=conf.n_mpi_process,  # the # of total main processes.
        # the # of subprocess for each main process.
        n_sub_process=conf.n_sub_process,
        comm_device=conf.comm_device,
        on_cuda=conf.on_cuda,
        rank=cur_rank,
    )
    conf.is_centralized = conf.graph_topology == "complete"

    # re-configure batch_size if sub_process > 1.
    if conf.n_sub_process > 1:
        conf.batch_size = conf.batch_size * conf.n_sub_process

    # configure cuda related.
    if conf.graph.on_cuda:
        assert torch.cuda.is_available()
        torch.manual_seed(conf.manual_seed)
        torch.cuda.manual_seed(conf.manual_seed)
        torch.cuda.set_device(conf.graph.device[0])
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True if conf.train_fast else False

    # define checkpoint for logging.
    checkpoint.init_checkpoint(conf)

    # configure logger.
    conf.logger = logging.Logger(conf.checkpoint_dir)
    conf.logger.log_metric = lm

    # display the arguments' info.
    logging.display_args(conf)


def init_distributed_world(conf, backend):
    if backend == "mpi":
        dist.init_process_group("mpi")
    elif backend == "nccl" or backend == "gloo":
        # init the process group.
        _tmp_path = os.path.join(conf.checkpoint, "tmp", conf.timestamp)
        op_paths.build_dirs(_tmp_path)

        dist_init_file = os.path.join(_tmp_path, "dist_init")

        torch.distributed.init_process_group(
            backend=backend,
            init_method="file://" + os.path.abspath(dist_init_file),
            timeout=datetime.timedelta(seconds=120),
            world_size=conf.n_mpi_process,
            rank=conf.rank,
        )
    else:
        raise NotImplementedError


def log_info(info_dict):
    """Add any information to MongoDB
       This function will be overwritten when called through run.py"""
    pass


def log_metric(name, values, tags={}):
    """Log timeseries data
       This function will be overwritten when called through run.py"""
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{key}:{value:7.3f}")
    values = ", ".join(value_list)
    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)
    print("{name:30s} - {values} ({tags})".format(name=name, values=values, tags=tags))


def lm(name, values, tags={}, display=False):
    log_metric(name, values, tags)
    if display:
        print("{name}: {values} ({tags})".format(name=name, values=values, tags=tags))


def log_runtime(label, mean_time, std, instances):
    """This function will be overwritten when called through run.py"""
    pass


def info(*args, **kwargs):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        log_info(*args, **kwargs)


if __name__ == "__main__":
    main()
