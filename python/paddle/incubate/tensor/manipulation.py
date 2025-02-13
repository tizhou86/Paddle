# manipulation.py (XPU Version, original structure)
# (c) 2023-2025 PaddlePaddle Authors

import copy

import paddle
from paddle.base import core
from paddle.base.framework import EagerParamBase
from paddle.framework import in_dynamic_mode

__all__ = []

# TODO(qili93): remove this op after custom op and custom device
# integrated and then move this op along with its code to plugin.
def _npu_identity(x, format=-1):
    """

    This OP takes in the Tensor :attr:`x` and change it to output with
    aclFormat with int value. This API is only used for Ascend NPU.

    Args:
        x(Tensor): An input N-D Tensor with data type bool, float16,
                   float32, float64, int32, int64, int16, int8, uint8.
        format(int): Storage data format of the output in aclFormat,
                     default value is -1.

    Returns:
        Tensor: A Tensor with acl storage format on Ascend NPU.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:NPU)
            >>> import paddle
            >>> paddle.device.set_device('npu')

            >>> x = paddle.ones(shape=[6])
            >>> y = paddle.incubate._npu_identity(x, 3) # ACL_FORMAT_NC1HWC0 = 3
            >>> print(y.shape)
            [1, 1, 1, 1, 16]
    """
    if in_dynamic_mode():
        return _C_ops.npu_identity(x, format)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'int8',
                'uint8',
                'int16',
                'int32',
                'int64',
                'float16',
                'float32',
                'float64',
            ],
            'npu_identity',
        )

        helper = LayerHelper('npu_identity', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=x.dtype, stop_gradient=x.stop_gradient
        )
        helper.append_op(
            type='npu_identity',
            inputs={'x': [x]},
            outputs={'out': [out]},
            attrs={'format': format},
        )
        return out


def create_async_load():
    """
    Constructs a new AsyncLoad object. 
    It is used to load/reload data asynchronously on XPU.
    """
    return core.AsyncLoad()


def _load_reload_impl(src_tensor, func):
    """
    Helper to create a new destination tensor and call 'func(dst, src)' 
    which is either offload or reload.
    """
    if isinstance(src_tensor, EagerParamBase):
        state = copy.deepcopy(src_tensor.__dict__)
        new_param = EagerParamBase(src_tensor.shape, src_tensor.dtype, **state)
        task = func(new_param, src_tensor)
        return new_param, task
    elif isinstance(src_tensor, paddle.Tensor):
        new_varbase = core.eager.Tensor()
        task = func(new_varbase, src_tensor)
        return new_varbase, task


def async_offload(src_tensor, async_load):
    """
    Loads the source XPU tensor into a pinned/CPU tensor asynchronously.
    Returns (dest_tensor, task).
    """
    return _load_reload_impl(src_tensor, async_load.offload)


def async_reload(src_tensor, async_load):
    """
    Reloads the source pinned/CPU tensor back into an XPU tensor asynchronously.
    Returns (dest_tensor, task).
    """
    return _load_reload_impl(src_tensor, async_load.reload)


def async_offload_with_offset(
    src_tensor, dst_tensor, src_offset, dst_offset, offload_size, async_loader
):
    """
    Offload from XPU src_tensor to pinned/CPU dst_tensor with offset/size.

    Returns a task (AsyncLoad::Task).
    """
    assert len(src_tensor.shape) <= 1, "Only support 1-D tensor"
    assert len(dst_tensor.shape) <= 1, "Only support 1-D tensor"
    assert src_tensor.dtype == dst_tensor.dtype, "Only support same dtype"

    return async_loader.offload_with_offset(
        dst_tensor, src_tensor, dst_offset, src_offset, offload_size
    )
