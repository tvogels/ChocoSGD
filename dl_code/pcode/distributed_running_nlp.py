# -*- coding: utf-8 -*-
from copy import deepcopy

import numpy as np
import torch

from pcode.utils.checkpoint import save_to_checkpoint
from pcode.utils.logging import display_training_stat, display_test_stat, dispaly_best_test_stat
from pcode.utils.stat_tracker import RuntimeTracker
from pcode.utils.timer import Timer
from pcode.utils.auxiliary import get_model_difference
import pcode.utils.error_handler as error_handler
from pcode.create_dataset import load_data_batch


# sys.excepthook = error_handler.global_except_hook


def train_and_validate(
    conf, model, criterion, scheduler, optimizer, metrics, data_loader, log_progress
):
    print("=>>>> start training and validation.\n")
    assert (
        optimizer.__class__.__name__ != "ParallelCHOCO"
    ), "NLP tasks right now do not support ParallelCHOCO based on multiprocessing (please use optimizer=parallel_choco_v instead)."

    # define runtime stat tracker and start the training.
    tracker_tr = RuntimeTracker(metrics_to_track=metrics.metric_names)

    # get the timer.
    timer = conf.timer

    total_bits_sent = 0

    # break until finish expected full epoch training.
    print("=>>>> enter the training.\n")
    while True:
        # init the hidden state.
        _hidden = (
            model.module.init_hidden(conf.batch_size)
            if "DataParallel" == model.__class__.__name__
            else model.init_hidden(conf.batch_size)
        )

        # configure local step.
        for idx, batch in enumerate(data_loader["train_loader"]):
            model.train()
            scheduler.step(optimizer)

            # repackage the hidden.
            _hidden = (
                model.module.repackage_hidden(_hidden)
                if "DataParallel" == model.__class__.__name__
                else model.repackage_hidden(_hidden)
            )

            # load data
            with timer("load_data", epoch=scheduler.epoch_):
                _input = batch.text[
                    :, conf.graph.rank * conf.batch_size : (conf.graph.rank + 1) * conf.batch_size
                ]
                _target = batch.target[
                    :, conf.graph.rank * conf.batch_size : (conf.graph.rank + 1) * conf.batch_size
                ]
                _input, _target = load_data_batch(conf, _input, _target)

            # inference and get current performance.
            with timer("forward_pass", epoch=scheduler.epoch_):
                optimizer.zero_grad()
                loss, _hidden = inference(
                    conf, model, criterion, metrics, _input, _target, _hidden, tracker_tr
                )
            with timer("backward_pass", epoch=scheduler.epoch_):
                loss.backward()

            with timer("sync_complete", epoch=scheduler.epoch_):
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf.rnn_clip)
                n_bits_to_transmit = optimizer.step(timer=timer)
                total_bits_sent += n_bits_to_transmit

            # print("bits", n_bits_to_transmit)

            # display the logging info.

            # finish one epoch training and to decide if we want to val our model.
            if scheduler.epoch_ % conf.eval_freq == 0:
                log_progress(scheduler.epoch_ / conf.num_epochs)

                display_training_stat(conf, scheduler, tracker_tr, total_bits_sent)

                if tracker_tr.stat["loss"].avg > 1e3 or np.isnan(tracker_tr.stat["loss"].avg):
                    print("\nThe process diverges!!!!!Early stop it.")
                    error_handler.abort()

                # each worker finish one epoch training.
                do_validate(
                    conf,
                    model,
                    optimizer,
                    criterion,
                    scheduler,
                    metrics,
                    data_loader,
                    total_bits_sent,
                )

                # refresh the logging cache at the begining of each epoch.
                tracker_tr.reset()

                # determine if the training is finished.
                if scheduler.is_stop():
                    conf.logger.save_json()
                    return

            # display tracking time.
            if (
                conf.graph.rank == 0
                and conf.display_tracked_time
                and scheduler.local_index % conf.summary_freq == 0
            ):
                print(timer.summary())


def inference(conf, model, criterion, metrics, _input, _target, _hidden, tracker=None):
    """Inference on the given model and get loss and accuracy."""
    output, _hidden = model(_input, _hidden)
    loss = criterion(output.view(-1, conf.n_tokens), _target.contiguous().view(-1))
    performance = metrics.evaluate(loss, output, _target)
    if tracker is not None:
        tracker.update_metrics([loss.item()] + performance, n_samples=_input.size(0))
    return loss, _hidden


def do_validate(
    conf, model, optimizer, criterion, scheduler, metrics, data_loader, total_bits_sent
):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # wait until the whole group enters this function, and then evaluate.
    performance = validate(
        conf, model, optimizer, criterion, scheduler, metrics, data_loader, total_bits_sent
    )

    # remember best performance and display the val info.
    scheduler.best_tracker.update(performance[0], scheduler.epoch_)
    dispaly_best_test_stat(conf, scheduler)

    # save to the checkpoint.
    save_to_checkpoint(
        conf,
        {
            "arch": conf.arch,
            "current_epoch": scheduler.epoch,
            "local_index": scheduler.local_index,
            "best_perf": scheduler.best_tracker.best_perf,
            "optimizer": optimizer.state_dict(),
            "state_dict": model.state_dict(),
        },
        scheduler.best_tracker.is_best,
        dirname=conf.checkpoint_dir,
        filename="checkpoint.pth.tar",
        save_all=conf.save_all_models,
    )
    print("Finished validation.")


def validate(conf, model, optimizer, criterion, scheduler, metrics, data_loader, total_bits_sent):
    """A function for model evaluation."""

    def _evaluate(_model, label):
        # define stat.
        tracker_te = RuntimeTracker(metrics_to_track=metrics.metric_names)

        # switch to evaluation mode
        _model.eval()

        # define hidden state for RNN.
        _hidden = (
            model.module.init_hidden(conf.batch_size)
            if "DataParallel" == model.__class__.__name__
            else model.init_hidden(conf.batch_size)
        )

        for batch in data_loader["val_loader"]:
            # load data and check performance.
            _input, _target = batch.text, batch.target

            # repackage the hidden.
            _hidden = (
                model.module.repackage_hidden(_hidden)
                if "DataParallel" == model.__class__.__name__
                else model.repackage_hidden(_hidden)
            )

            with torch.no_grad():
                _, _hidden = inference(
                    conf, _model, criterion, metrics, _input, _target, _hidden, tracker_te
                )

        # display the test stat.
        display_test_stat(conf, scheduler, tracker_te, label, bits=total_bits_sent)

        # get global (mean) performance
        global_performance = tracker_te.evaluate_global_metrics()
        return global_performance

    # # evaluate the averaged local model on the validation dataset.
    # if (
    #     conf.graph_topology != "complete"
    #     and conf.graph_topology != "data_center"
    #     and not conf.train_fast
    # ):
    #     copied_model = deepcopy(model)
    #     optimizer.world_aggregator.agg_model(copied_model, op="avg")
    #     _evaluate(copied_model, label="averaged_model")

    #     # get the l2 distance of the local model to the averaged model
    #     conf.logger.log_metric(
    #         name="stat",
    #         values={
    #             "rank": conf.graph.rank,
    #             "epoch": scheduler.epoch_,
    #             "distance": get_model_difference(model, copied_model),
    #         },
    #         tags={"split": "test", "type": "averaged_model"},
    #     )

    # evaluate each local model on the validation dataset.
    global_performance = _evaluate(model, label="local_model")
    return global_performance
