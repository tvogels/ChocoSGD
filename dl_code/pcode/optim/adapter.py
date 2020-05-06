import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
import thijs_code.algorithms as algorithms
import thijs_code.compressors as compressors
import thijs_code.topologies as topologies

"""Adapts Thijs' optimizer design to Tao's signature"""


class Adapter(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        conf=None,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)
        self.conf = conf

        # define reducer.
        self.backend = conf.backend

        # define sorted param names.
        self.param_names = list(enumerate([group["name"] for group in self.param_groups]))

        update_fn = lambda: utils.apply_gradient(
            self.param_groups, self.state, apply_grad_to_model=True
        )

        self.optimizer = get_optimizer(
            vars(conf),
            conf.timer,
            TopologyAdapter(conf),
            sum([group["params"] for group in params], []),
            update_fn,
        )

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        assert closure is None

        grads, _ = comm.get_data(self.param_groups, self.param_names, is_get_grad=True)

        bits_before = self.optimizer.bits_sent

        self.optimizer.step(lambda: (None, grads, None))

        bits_after = self.optimizer.bits_sent

        return bits_after - bits_before


class TopologyAdapter(topologies.Topology):
    def __init__(self, conf):
        super().__init__(conf.graph.n_nodes)
        self.diffusion_matrix = conf.graph._mixing_matrix
        if not isinstance(self.diffusion_matrix, np.ndarray):
            self.diffusion_matrix = self.diffusion_matrix.todense()
        self.name = type(conf.graph).__name__


def get_optimizer(config, timer, topology, params, update_fn):
    if config["optimizer"] == "dpsgd":
        return algorithms.DPSGD(timer, algorithms.SimpleGossip(topology), params, update_fn)
    elif config["optimizer"] == "d2":
        assert config["momentum"] == 0, "exact diffusion doesn't support momentum"
        return algorithms.D2(
            timer, algorithms.SimpleGossip(topology), params, config["learning_rate"]
        )
    elif config["optimizer"] == "exact-diffusion":
        assert config["momentum"] == 0, "exact diffusion doesn't support momentum"
        return algorithms.ExactDiffusion(
            timer, algorithms.SimpleGossip(topology), params, update_fn
        )
    elif config["optimizer"] == "scaffold":
        assert config["momentum"] == 0, "scaffold doesn't support momentum"
        return algorithms.Scaffold(
            timer,
            algorithms.SimpleGossip(topology),
            params,
            learning_rate=config["learning_rate"],
            gossip_after_update=config["scaffold_gossip_after_update"],
        )
    elif config["optimizer"] == "all-reduce":
        return algorithms.DPSGD(
            timer, algorithms.AllReduce(topology), params, update_fn, overlapping=False
        )
    elif config["optimizer"] == "power-gossip":
        return algorithms.DPSGD(
            timer,
            algorithms.OnlyOnLargeParameters(
                topology,
                algorithms.PowerGossip(
                    topology,
                    rank=config["optimizer_rank"],
                    num_iterations=config["optimizer_num_iterations"],
                    warm_start=config["optimizer_warm_start"],
                ),
            ),
            params,
            update_fn,
        )
    elif config["optimizer"] == "d2-power-gossip":
        assert config["momentum"] == 0, "exact diffusion doesn't support momentum"
        return algorithms.D2(
            timer,
            algorithms.OnlyOnLargeParameters(
                topology,
                algorithms.PowerGossip(
                    topology,
                    rank=config["optimizer_rank"],
                    num_iterations=config["optimizer_num_iterations"],
                    warm_start=config["optimizer_warm_start"],
                ),
            ),
            params,
            config["learning_rate"],
        )
    elif config["optimizer"] == "exact-diffusion-power-gossip":
        assert config["momentum"] == 0, "exact diffusion doesn't support momentum"
        return algorithms.ExactDiffusion(
            timer,
            algorithms.OnlyOnLargeParameters(
                topology,
                algorithms.PowerGossip(
                    topology,
                    rank=config["optimizer_rank"],
                    num_iterations=config["optimizer_num_iterations"],
                    warm_start=config["optimizer_warm_start"],
                ),
            ),
            params,
            update_fn,
        )
    elif config["optimizer"] == "scaffold-power-gossip":
        assert config["momentum"] == 0, "scaffold doesn't support momentum"
        return algorithms.Scaffold(
            timer,
            algorithms.OnlyOnLargeParameters(
                topology,
                algorithms.PowerGossip(
                    topology,
                    rank=config["optimizer_rank"],
                    num_iterations=config["optimizer_num_iterations"],
                    warm_start=config["optimizer_warm_start"],
                ),
            ),
            params,
            learning_rate=config["learning_rate"],
            gossip_after_update=config["scaffold_gossip_after_update"],
        )
    elif config["optimizer"] == "choco" or config["optimizer"] == "deepsqueeze":
        classType = {"choco": algorithms.ChocoGossip, "deepsqueeze": algorithms.DeepSqueezeGossip}[
            config["optimizer"]
        ]

        if config["optimizer_compressor"] == "top-k":
            compressor = compressors.TopK(rank=config["optimizer_rank"])
        elif config["optimizer_compressor"] == "svd":
            compressor = compressors.SVD(rank=config["optimizer_rank"])
        elif config["optimizer_compressor"] == "sign-and-norm":
            compressor = compressors.SignAndNorm()
        else:
            raise ValueError("Unknown compressor {}".format(config["optimizer_compressor"]))

        return algorithms.DPSGD(
            timer,
            classType(
                topology, diffusion_rate=config["optimizer_diffusion_rate"], compressor=compressor
            ),
            params,
            update_fn,
        )
    elif config["optimizer"] == "moniqua":
        return algorithms.DPSGD(
            timer,
            algorithms.MoniquaGossip(
                topology,
                diffusion_rate=config["optimizer_diffusion_rate"],
                theta=config["optimizer_theta"],
            ),
            params,
            update_fn,
        )
    else:
        raise ValueError("Unknown optimizer {}".format(config["optimizer"]))
