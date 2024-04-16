

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
from typing import Optional, Any
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.module import Module


class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""

    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        # if self.track_running_stats:
        #     self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
        #     self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
        #     self.running_mean: Optional[Tensor]
        #     self.running_var: Optional[Tensor]
        #     self.register_buffer('num_batches_tracked',
        #                          torch.tensor(0, dtype=torch.long,
        #                                       **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        # else:
        #     self.register_buffer("running_mean", None)
        #     self.register_buffer("running_var", None)
        #     self.register_buffer("num_batches_tracked", None)
        # self.reset_parameters()
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.register_buffer('running_mean_target', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var_target', torch.ones(num_features, **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.running_mean_target: Optional[Tensor]
            self.running_var_target: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("running_mean_target", None)
            self.register_buffer("running_var_target", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()
    #
    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.running_mean_target.zero_()  # type: ignore[union-attr]
            self.running_var_target.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]
            # self.running_mean.zero_()  # type: ignore[union-attr]
            # self.running_var.fill_(1)  # type: ignore[union-attr]
            # self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]
    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_NormBase, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if self.training:
            batch_size = input.size()[0] // 2
            input_source = input[:batch_size]
            input_target = input[batch_size:]
            z_source = F.batch_norm(
                input_source,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean
                if not self.training or self.track_running_stats
                else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,)
            z_target = F.batch_norm(
                input_target,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean_target
                if not self.training or self.track_running_stats
                else None,
                self.running_var_target if not self.training or self.track_running_stats else None,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,)
            z = torch.cat((z_source, z_target), dim=0)

            if input.dim() == 4:    ## TransNorm2d
                input_source = input_source.permute(0,2,3,1).contiguous().view(-1,self.num_features)
                input_target = input_target.permute(0,2,3,1).contiguous().view(-1,self.num_features)

            cur_mean_source = torch.mean(input_source, dim=0)
            cur_var_source = torch.var(input_source,dim=0)
            cur_mean_target = torch.mean(input_target, dim=0)
            cur_var_target = torch.var(input_target, dim=0)

            ## 3. Domain Adaptive Alpha.

            ### 3.1 Calculating Distance
            dis = torch.abs(cur_mean_source / torch.sqrt(cur_var_source + self.eps) -
                            cur_mean_target / torch.sqrt(cur_var_target + self.eps))

            ### 3.2 Generating Probability
            prob = 1.0 / (1.0 + dis)
            # print(prob)
            alpha = self.num_features * prob / sum(prob)

            if input.dim() == 2:
                alpha = alpha.view(1, self.num_features)
            elif input.dim() == 4:
                alpha = alpha.view(1, self.num_features, 1, 1)
            return z * (1 + alpha.detach())
        else:
            assert bn_training==False
            # print('===================================================>测试阶段')
            # bn_training = (self.running_mean_target is None) and (self.running_var_target is None)
            z = F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean_target
            if not self.training or self.track_running_stats
            else None,
            self.running_var_target if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,)

            dis = torch.abs(self.running_mean / torch.sqrt(self.running_var + self.eps)
                               - self.running_mean_target / torch.sqrt(self.running_var_target + self.eps))
            prob = 1.0 / (1.0 + dis)
            # print(prob)
            alpha = self.num_features * prob / sum(prob)

            if input.dim() == 2:
                alpha = alpha.view(1, self.num_features)
            elif input.dim() == 4:
                alpha = alpha.view(1, self.num_features, 1, 1)
            return z * (1 + alpha.detach())

class BatchNorm2d(_BatchNorm):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

