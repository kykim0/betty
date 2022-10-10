# Copyright Sang Keun Choe
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

import torch
import torch.distributed as dist

from betty.configs import Config
from betty.hypergradient import get_grads
from betty.utils import convert_tensor, log_from_loss_dict


class Problem:
    """
    This is the base class for an optimization problem in multilevel optimization.
    Specifically, each problem is defined by the parameter (or module), the sets of the upper
    and lower constraining problems, the dataset, the loss function, the optimizer, and other
    optimization configurations (e.g. best-response Jacobian calculation algorithm, number of
    unrolling steps, etc.).
    """

    def __init__(
        self,
        name=None,
        config=None,
        module=None,
        optimizer=None,
        scheduler=None,
        train_data_loader=None,
        device=None,
    ):
        # basic configurations
        self._name_orig = name
        self._name = name if name is not None else "problem"
        self._config = config if config is not None else Config()
        self.engine_config = None

        # backend
        self._backend = None

        # device
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # distributed
        self._distributed = False

        # computation graph depedency
        self._parents = []
        self._children = []
        self._paths = []
        self._problem_name_dict = {}

        # data loader
        self.train_data_loader = train_data_loader
        self.train_data_iterator = None
        self.cur_batch = None
        self.epoch_counter = None

        # module
        self.module = module

        # optimizer & lr scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        # environment
        self.env = None

        # fp16 scaler
        self._fp16 = config.fp16
        self.scaler = None
        if self._fp16:
            self.initial_dynamic_scale = config.initial_dynamic_scale
            self.scale_factor = config.scale_factor

        # gradient accumulation
        self.gas = config.gradient_accumulation

        # gradient clipping
        self.gradient_clipping = config.gradient_clipping

        # warmup
        self.warmup_steps = config.warmup_steps

        # logger
        self.logger = None
        self.log_step = config.log_step
        self.log_local_step = config.log_local_step

        # step counter
        self._count = 0
        self._global_step = 0

        # misc
        self._leaf = False
        self._default_grad = False
        self._first_order = False
        self._retain_graph = config.retain_graph
        self._allow_unused = config.allow_unused
        self._unroll_steps = config.unroll_steps
        self._roll_back = False
        self._inner_loop_start = True
        self._training = True
        self.ready = None

    def _is_default_fp16(self):
        if not self._fp16 or self._backend in ["deepspeed", "accelerate"]:
            return False
        return True

    def parse_engine_config(self, engine_config):
        """Parse engine config and set global attributes accordingly.

        Args:
            engine_config: EngineConfig
        """
        self._backend = engine_config.backend

    def initialize(self, engine_config):
        """
        ``initialize`` patches/sets up module, optimizer, data loader, etc. after compiling a
        user-provided configuration (e.g., fp16 training, iterative differentiation)
        """
        if self._name_orig is None:
            self.logger.warning(
                "name is not defined for this Problem. We arbitrarily set "
                "the name as 'problem' to avoid undesired behaviors."
            )
        delattr(self, "_name_orig")
        # engine config
        self.engine_config = engine_config

        # initialize update ready to False
        if self._leaf:
            assert len(self._children) == 0
        if len(self._paths) == 0:
            self._default_grad = True
        self.ready = [False for _ in range(len(self._children))]

        # compile parents configurations
        first_order = []
        for problem in self._parents:
            parent_config = problem.config
            first_order.append(parent_config.first_order)
        self._first_order = all(first_order)
        if len(self._parents) > 0:
            self._roll_back = engine_config.roll_back

        # compile children configurations
        children_unroll_steps = []
        for problem in self._children:
            child_config = problem.config
            children_unroll_steps.append(child_config.unroll_steps)

        self._inner_loop_start = True

        # set up data loader
        if self.train_data_loader is None and self.is_implemented(
            "configure_train_data_loader"
        ):
            self.train_data_loader = self.configure_train_data_loader()
        if self.train_data_loader is not None:
            if not isinstance(self.train_data_loader, tuple):
                self.train_data_loader = (self.train_data_loader,)

            self.train_data_iterator = []
            self.epoch_counter = []
            for train_data_loader in self.train_data_loader:
                if self._distributed:
                    train_data_loader.sampler.set_epoch(0)
                self.train_data_iterator.append(iter(train_data_loader))
                self.epoch_counter.append(0)
        else:
            assert self.is_implemented("get_batch")

        # set up module for the current level
        if self.is_implemented("configure_module"):
            if self.configure_module() is not None:
                self.module = self.configure_module()
        assert self.module is not None, "Module must be specified!"
        self.module.to(self.device)

        # set up optimizer
        if self.is_implemented("configure_optimizer"):
            if self.configure_optimizer() is not None:
                self.optimizer = self.configure_optimizer()

        # set up lr scheduler
        if self.is_implemented("configure_scheduler"):
            if self.configure_scheduler is not None:
                self.scheduler = self.configure_scheduler()

        # set up fp16 training
        if self._is_default_fp16():
            assert torch.cuda.is_available()
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=self.initial_dynamic_scale, growth_factor=self.scale_factor
            )

        # Logging INFO
        # TODO: Replace print with logging
        path_str = [[node.name for node in path] for path in self._paths]
        children_str = [node.name for node in self._children]
        parents_str = [node.name for node in self._parents]
        self.logger.info("*** Problem Information ***")
        self.logger.info(f"Name: {self._name}")
        self.logger.info(f"Uppers: {parents_str}")
        self.logger.info(f"Lowers: {children_str}")
        self.logger.info(f"Paths: {path_str}\n")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Users define how forward (or call) function is defined for the problem here.
        """
        return self.module(*args, **kwargs)

    @abc.abstractmethod
    def training_step(self, batch):
        """
        Users define the loss function of the problem here.
        """
        raise NotImplementedError

    def training_step_exec(self, batch):
        if self._is_default_fp16():
            with torch.cuda.amp.autocast():
                return self.training_step(batch)
        else:
            return self.training_step(batch)

    def one_step_descent(self, batch=None):
        # load data
        if batch is None:
            self.cur_batch = self.get_batch()
            batch = self.cur_batch

        # calculate loss
        loss, loss_dict = self.get_loss(batch)

        # calculate gradient (a.k.a backward)
        self.backward(
            loss=loss,
            params=self.trainable_parameters(),
            paths=self._paths,
            create_graph=not self._first_order,
            retain_graph=self._retain_graph,
            allow_unused=self._allow_unused,
        )

        # calculate parameter update
        if self._count % self.gas == 0:
            if self._distributed:
                self.synchronize_grads(self.trainable_parameters)
            self.optimizer_step()

            # param callback (e.g., parameter clipping)
            if self.is_implemented("param_callback"):
                self.param_callback(self.trainable_parameters())

            # zero-out grad
            self.zero_grad()

        return loss_dict

    def step_normal(self, global_step=None):
        if self.check_ready():
            # loop start
            if self._inner_loop_start:
                if self.is_implemented("on_inner_loop_start"):
                    self.on_inner_loop_start()
                self._inner_loop_start = False

                # copy current parameters, buffers, optimizer states
                if self._roll_back:
                    self.cache_states()

            # increase count (local step)
            if self._training:
                self._count += 1

            # one step grdient descent
            loss_dict = self.one_step_descent()

            # lr scheduler step
            if self.scheduler is not None and not self._roll_back:
                self.scheduler.step()

            # logging
            if self.log_step > 0 and self._count % self.log_step == 0:
                self.log(loss_dict, global_step)

            # call parent step_normal after unrolling
            if (
                self._training
                and self._count % (self._unroll_steps * self.gas) == 0
                and self._count > self.warmup_steps
            ):
                for problem in self._parents:
                    idx = problem.children.index(self)
                    problem.ready[idx] = True
                    problem.step_normal(global_step=global_step)

                self._inner_loop_start = True

            self.ready = [False for _ in range(len(self._children))]

    def step_after_roll_back(self):
        if self.check_ready() and self._training:
            if self._roll_back:
                # recover from cached states
                self.recover_states()

                # one step gradient step
                _ = self.one_step_descent(batch=self.cur_batch)

                # lr scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                # call parent step_after_roll_back
                for problem in self._parents:
                    idx = problem.children.index(self)
                    problem.ready[idx] = True
                    problem.step_after_roll_back()

            self.ready = [False for _ in range(len(self._children))]

    def step(self, global_step=None):
        """
        ``step`` method abstracts a one-step gradient descent update with four sub-steps:
        1) data loading, 2) cost calculation, 3) gradient calculation, and 4) parameter update.
        It also calls upper-level problems' step methods after unrolling gradient steps based on
        the hierarchical dependency graph.

        :param global_step: global step of the whole multilevel optimization. Defaults to None.
        :type global_step: int, optional
        """
        self._global_step = global_step
        self.step_normal(global_step=global_step)
        if (
            self._count % (self._unroll_steps * self.gas) == 0
            and self._count > self.warmup_steps
        ):
            self.step_after_roll_back()

    def get_batch(self):
        """
        Load training batch from the user-provided data loader

        :return: New training batch
        :rtype: Any
        """
        batch = tuple(
            self.get_batch_single_loader(i) for i in range(len(self.train_data_loader))
        )

        return batch[0] if len(batch) == 1 else batch

    def get_batch_single_loader(self, idx):
        """
        Load training batch from one of the user-provided data loader(s)

        :return: New training batch
        :rtype: Any
        """
        data_iterator = self.train_data_iterator[idx]
        try:
            batch = next(data_iterator)
        except StopIteration:
            if idx == 0:
                self.on_epoch_end_exec()
            self.epoch_counter[idx] += 1
            train_data_loader = self.train_data_loader[idx]
            if self._distributed:
                train_data_loader.sampler.set_epoch(self.epoch_counter[idx])
            self.train_data_iterator[idx] = iter(train_data_loader)
            batch = next(self.train_data_iterator[idx])
        if not isinstance(batch, dict):
            batch = tuple(
                convert_tensor(value, self.device, self._is_default_fp16())
                for value in batch
            )
        else:
            for key, value in batch.items():
                batch[key] = convert_tensor(value, self.device, self._is_default_fp16())

        return batch

    def get_loss(self, batch):
        """
        Calculate loss and log metrics for the current batch based on the user-defined loss
        function.

        :return: loss and log metrics (e.g. classification accuracy)
        :rtype: dict
        """
        maybe_loss_dict = self.training_step_exec(batch)
        is_dict = isinstance(maybe_loss_dict, dict)
        loss = maybe_loss_dict["loss"] if is_dict else maybe_loss_dict
        loss_no_scale = loss.item()
        if self._is_default_fp16():
            loss = self.scaler.scale(loss)
        loss = loss / self.gas

        # construct loss dict
        loss_dict = {"loss": loss_no_scale}
        if is_dict:
            for key, value in maybe_loss_dict.items():
                if key != "loss":
                    loss_dict[key] = value

        return loss, loss_dict

    def backward(
        self,
        loss,
        params,
        paths,
        create_graph=False,
        retain_graph=True,
        allow_unused=True,
    ):
        """
        Calculate the gradient of ``loss`` with respect to ``params`` based on a user-defined
        ``config``.

        :param loss: Outputs of the differentiated function.
        :type loss: Tensor
        :param params: Inputs with respect to which the gradient will be returned.
        :type params: Sequence of Tensor
        :param paths: Paths on which the gradient will be calculated.
        :type paths: List of list of Problem
        :param create_graph:
            If ``True``, graph of the derivative will be constructed, allowing to compute higher order
            derivative products. Default: ``True``.
        :type create_graph: bool, optional
        :param retain_graph:
            If ``False``, the graph used to compute the grad will be freed. Note that in nearly all
            cases setting this option to ``True`` is not needed and often can be worked around in a much
            more efficient way. Defaults to the value of ``create_graph``.
        :type retain_graph: bool, optional
        :param allow_unused:
            If ``False``, specifying inputs that were not used when computing outputs (and therefore
            their grad is always zero) is an error. Defaults to ``False``.
        :type allow_unused: bool, optional
        """
        # direct grad
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=allow_unused,
        )
        self.set_grads(params, grads)

        # indirect grad: best-response Jacobian
        if self._config.first_order:
            for idx, path in enumerate(paths):
                retain_graph_implicit = False if idx == len(paths) - 1 else True
                grads = get_grads(loss, path, retain_graph_implicit)
                self.set_grads(params, grads)

    def set_grads(self, params, grads):
        """
        Set gradients for trainable parameters. ``params.grad = grads``

        :param params: Trainable parameters
        :type params: Sequence of Tensor
        :param grads: Calculated gradient
        :type grads: Sequence of Tensor
        """
        for param, grad in zip(params, grads):
            if grad is not None:
                if hasattr(param, "grad") and param.grad is not None:
                    param.grad = param.grad + grad
                else:
                    param.grad = grad

    def synchronize_grads(self, params):
        for param in params:
            grad = param.grad

            # perform all_reduce
            grad.mul_(1.0 / dist.get_world_size())
            dist.all_reduce(grad)

            #! May not need this line
            param.grad.copy_(grad)

    @abc.abstractmethod
    def optimizer_step(self, *args, **kwargs):
        """
        Update weights as in PyTorch's native ``optim.step()``
        """
        raise NotImplementedError

    def zero_grad(self):
        """
        Set gradients for trainable parameters for the current problem to 0.
        Similar with PyTorch's ``optim.zero_grad()`` or ``module.zero_grad()``.
        """
        for param in list(self.trainable_parameters()):
            if hasattr(param, "grad"):
                del param.grad

    def clip_grad(self):
        torch.nn.utils.clip_grad_norm_(
            parameters=self.trainable_parameters(), max_norm=self.gradient_clipping
        )

    def state_dict(self):
        """
        Return all states involved in ``Problem`` with a Python dictionary. By default, it
        includes ``self.module.state_dict`` and ``self.optimizer.state_dict``. Depending on users'
        configurations, it may include ``self.scheuler.state_dict`` (lr scheduler) and
        ``self.scaler.state_dict`` (fp16 training)
        """
        state_dict = {}
        state_dict["module"] = self.module.state_dict()
        state_dict["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        if self._is_default_fp16():
            state_dict["scaler"] = self.scaler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Load the state for the ``Problem``

        Args:
            state_dict (dict): Python dictionary of Problem states
        """
        self.module.load_state_dict(state_dict["module"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.scheduler is not None and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        if self._is_default_fp16() and "scaler" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler"])

    @abc.abstractmethod
    def cache_states(self):
        """
        Cache params, buffers, optimizer states when ``config.roll_back`` is set to ``True`` in
        ``step``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recover_states(self):
        """
        Recover params, buffers, optimizer states when ``config.roll_back`` is set to ``True`` in
        ``step``.
        """
        raise NotImplementedError

    def on_epoch_end_exec(self):
        if self.is_implemented("on_epoch_end"):
            self.on_epoch_end()

    def is_implemented(self, fn_name):
        """
        Check if ``fn_name`` method is implemented in the class

        :rtype: bool
        """
        return callable(getattr(self, fn_name, None))

    def check_ready(self):
        """
        Check if unrolling processes of lower level problems in the hierarchical dependency
        graph are all ready/done. ``step`` function is only excuted when this method returns
        ``True``.

        :rtype: bool
        """
        return all(self.ready)

    def log(self, stats, global_step):
        """
        Log (training) stats to the ``self.logger``

        :param stats: log metrics such as loss and classification accuracy.
        :type stats: Any
        :param step: global/local step associated with the ``stats``.
        :type step: int
        """
        loss_log = log_from_loss_dict(stats)
        if global_step is None:
            self.logger.info(
                f'[Problem "{self._name}"] [Local Step {self._count}] {loss_log}'
            )
        else:
            self.logger.info(
                f'[Problem "{self._name}"] [Global Step {global_step}] [Local Step {self._count}] '
                f"{loss_log}"
            )
        cur_step = global_step
        if global_step is None or self.log_local_step:
            cur_step = self._count
        self.logger.log(stats, tag=self._name, step=cur_step)

    def set_problem_attr(self, problem):
        """
        Set class attributes for upper-/lower-level problems based on their names.

        :param problem: lower- or upper-level problem in the dependency graph
        :type problem: Problem
        :return: name of ``problem``
        :rtype: str
        """
        name = problem.name
        if name not in self._problem_name_dict:
            assert not hasattr(
                self, name
            ), f"Problem already has an attribute named {name}!"
            self._problem_name_dict[name] = 0
            setattr(self, name, problem)
        elif self._problem_name_dict[name] == 0:
            # rename first problem
            first_problem = getattr(self, name)
            delattr(self, name)
            setattr(self, name + "_0", first_problem)

            self._problem_name_dict[name] += 1
            name = name + "_" + str(self._problem_name_dict[name])
            setattr(self, name, problem)
        else:
            self._problem_name_dict[name] += 1
            name = name + "_" + str(self._problem_name_dict[name])
            setattr(self, name, problem)

        return name

    def add_child(self, problem):
        """
        Add ``problem`` to the lower-level problem list.

        :param problem: lower-level problem in the dependency graph
        :type problem: Problem
        """
        assert problem not in self._children
        self._children.append(problem)

    def add_parent(self, problem):
        """
        Add ``problem`` to the upper-level problem list.

        :param problem: upper-level problem in the dependency graph
        :type problem: Problem
        """
        assert problem not in self._parents
        self._parents.append(problem)

    def add_paths(self, paths):
        """
        Add new hypergradient backpropagation paths.
        """
        self._paths.extend(paths)

    def add_logger(self, logger):
        """
        Add logger to the current problem.

        :param logger: logger defined by users in ``Engine``.
        """
        if self.logger is None:
            self.logger = logger

    def add_env(self, env):
        """
        Add environment to the current problem.

        :param env: Environment.
        """
        if self.env is None:
            self.env = env

    @abc.abstractmethod
    def parameters(self):
        """
        Return all parameters for the current problem.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def trainable_parameters(self):
        """
        Define all *trainable* parameters for the current problem.
        """
        raise NotImplementedError

    def clear_dependencies(self):
        """
        Clear the dependencies of the current problem.
        """
        self._children = []
        self._parents = []
        self._paths = []

    def train(self):
        """
        Set the current problem to the training mode.
        """
        self._training = True

    def eval(self):
        """
        Set the current problem to the evaluation mode.
        """
        self._training = False

    @property
    def name(self):
        """[summary]
        Return the user-defined name of the module.
        """
        return self._name

    @property
    def config(self):
        """
        Return the configuration for the current problem.
        """
        return self._config

    @property
    def children(self):
        """
        Return lower-level problems for the current problem.
        """
        return self._children

    @property
    def parents(self):
        """
        Return upper-level problems for the current problem.
        """
        return self._parents

    @property
    def paths(self):
        """
        Return hypergradient calculation paths for the current problem.
        """
        return self._paths

    @property
    def leaf(self):
        """
        Return whether the current problem is leaf or not.

        :return: leaf
        :rtype: bool
        """
        return self._leaf

    @property
    def count(self):
        """
        Return the local step for the current problem.

        :return: local step
        :rtype: int
        """
        return self._count

    @leaf.setter
    def leaf(self, leaf):
        """
        Set the current problem as a leaf problem.
        """
        self._leaf = leaf

    @property
    def distributed(self):
        """
        Return whether distributed training is enabled.

        :return: distributed
        :rtype: bool
        """
        return self._distributed

    @distributed.setter
    def distributed(self, distributed):
        """
        Set the distributed training status.
        """
        self._distributed = distributed
