#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
# Force the "spawn" start method BEFORE importing paddle or anything that might call CUDA/XPU APIs.
multiprocessing.set_start_method("spawn", force=True)

import gc
import os
import time
import unittest

import paddle
import paddle.incubate.multiprocessing as mp

REPEAT = 20
HAS_SHM_FILES = os.path.isdir('/dev/shm')

# ------------------------------------------------------------------------------
# Helper functions for XPU IPC sharing with extra debugging output
# ------------------------------------------------------------------------------
def maybe_share_xpu(tensor):
    """
    If the tensor is on XPU, return its IPC metadata (a tuple) so that
    it can be sent safely via a multiprocessing.Queue.
    """
    if tensor.place.is_xpu_place():
        ipc_meta = tensor.value().get_tensor()._share_xpu()
        print("[DEBUG] maybe_share_xpu: tensor on XPU, ipc_meta type:",
              type(ipc_meta), "value:", ipc_meta)
        return ipc_meta
    print("[DEBUG] maybe_share_xpu: tensor not on XPU, returning original tensor")
    return tensor

def maybe_unshare_xpu(x):
    """
    If x is not a paddle.Tensor (i.e. it is IPC metadata), re-create the
    shared tensor from the metadata. Use .shape if dims() is unavailable.
    """
    if not isinstance(x, paddle.Tensor):
        print("[DEBUG] maybe_unshare_xpu: received IPC meta of type:",
              type(x), "value:", x)
        try:
            shared_obj = paddle.base.core.LoDTensor._new_shared_xpu(x)
            shape_info = getattr(shared_obj, "shape", None)
            if shape_info is None:
                shape_info = getattr(shared_obj, "dims", None)
            print("[DEBUG] maybe_unshare_xpu: new shared XPU object shape:", shape_info)
            new_tensor = paddle.to_tensor(shared_obj)
            print("[DEBUG] maybe_unshare_xpu: converted to paddle.Tensor with place:",
                  new_tensor.place)
            return new_tensor
        except Exception as e:
            print("[DEBUG] maybe_unshare_xpu: Exception occurred:", e)
            raise
    else:
        print("[DEBUG] maybe_unshare_xpu: received a paddle.Tensor, returning unchanged")
    return x

# ------------------------------------------------------------------------------
# Child Process Initialization for XPU
# ------------------------------------------------------------------------------
def init_xpu_child():
    """Ensure that the child process sets the XPU device correctly."""
    try:
        paddle.set_device("xpu")
        # Print the current device for debugging.
        current_device = paddle.get_device() if hasattr(paddle, "get_device") else "xpu"
        print("[DEBUG Child] XPU device set in child process; current device:", current_device)
    except Exception as e:
        print("[DEBUG Child] Exception when setting XPU device:", e)

# ------------------------------------------------------------------------------
# Worker functions with additional debugging prints (to be run in child processes)
# ------------------------------------------------------------------------------
def fill_tensor(queue, event):
    init_xpu_child()
    data = queue.get()
    print("[DEBUG] fill_tensor: received data from queue:", data)
    data = [maybe_unshare_xpu(t) for t in data]
    print("[DEBUG] fill_tensor: after maybe_unshare_xpu, data[0] type:",
          type(data[0]), "place:", data[0].place)
    try:
        print("Child: Before modification, data[0] =", data[0].numpy())
        if len(data) > 1:
            print("Child: Before modification, data[1] =", data[1].numpy())
    except Exception as e:
        print("Child: Unable to print tensor before modification:", e)
    with paddle.no_grad():
        try:
            data[0][:] = 5
        except Exception as e:
            print("[DEBUG] fill_tensor: Exception during data[0][:] = 5:", e)
            raise
        if len(data) > 1:
            data[1][:] = 5
    try:
        print("Child: After modification, data[0] =", data[0].numpy())
        if len(data) > 1:
            print("Child: After modification, data[1] =", data[1].numpy())
    except Exception as e:
        print("Child: Unable to print tensor after modification:", e)
    event.set()

def send_tensor(queue, event, device, dtype):
    init_xpu_child()
    tensor = paddle.ones([5, 5], dtype=dtype)
    print("[DEBUG] send_tensor: created tensor:", tensor.numpy())
    if device.lower() == "xpu":
        tensor = maybe_share_xpu(tensor)
        print("[DEBUG] send_tensor: shared tensor ipc_meta:", tensor)
    queue.put(tensor)
    queue.put(tensor)
    event.wait()

def send_parambase(queue, event, device, dtype):
    init_xpu_child()
    tensor = paddle.nn.Layer().create_parameter(
        [5, 5],
        dtype=dtype,
        default_initializer=paddle.nn.initializer.Constant(value=1.0),
    )
    print("[DEBUG] send_parambase: created parameter tensor:", tensor.numpy())
    if device.lower() == "xpu":
        tensor_ipc = maybe_share_xpu(tensor)
        slice_ipc = maybe_share_xpu(tensor[:, 1].detach())
        print("[DEBUG] send_parambase: tensor_ipc:", tensor_ipc)
        print("[DEBUG] send_parambase: slice_ipc:", slice_ipc)
        queue.put(tensor_ipc)
        queue.put(slice_ipc)
    else:
        queue.put(tensor)
        queue.put(tensor[:, 1].detach())
    event.wait()

def check_ipc_tensor_xpu(event, ipc_metas):
    init_xpu_child()
    print("[DEBUG] check_ipc_tensor_xpu: received ipc_metas:", ipc_metas)
    ground_truth1 = paddle.to_tensor([1, 2, 3]).to("xpu").astype("int32")
    ground_truth2 = paddle.to_tensor([3, 4, 5]).to("xpu").astype("int32")
    try:
        shared_ipc_tensor = paddle.to_tensor(
            paddle.base.core.LoDTensor._new_shared_xpu(ipc_metas)
        )
    except Exception as e:
        print("[DEBUG] check_ipc_tensor_xpu: Exception during _new_shared_xpu:", e)
        raise
    print("[DEBUG] check_ipc_tensor_xpu: created shared_ipc_tensor, place:",
          shared_ipc_tensor.place, "dims:", shared_ipc_tensor.shape)
    shared_ipc_tensor_cpu = shared_ipc_tensor.cpu().astype("int32")
    print("Debug: shared_ipc_tensor_cpu place =", shared_ipc_tensor_cpu.place)
    print("Debug: shared_ipc_tensor_cpu shape =", shared_ipc_tensor_cpu.shape)
    print("Debug: shared_ipc_tensor_cpu dtype =", shared_ipc_tensor_cpu.dtype)
    print("Debug: ground_truth1_cpu: shape={}, dtype={}, place={}".format(
        ground_truth1.cpu().shape, ground_truth1.cpu().dtype, ground_truth1.cpu().place))
    print("Debug: ground_truth2_cpu: shape={}, dtype={}, place={}".format(
        ground_truth2.cpu().shape, ground_truth2.cpu().dtype, ground_truth2.cpu().place))
    def tensor_equal(t1, t2):
        try:
            return (t1 == t2).all().item()
        except Exception as e:
            print("Comparison error:", e)
            print("t1: shape={}, dtype={}, place={}".format(t1.shape, t1.dtype, t1.place))
            print("t2: shape={}, dtype={}, place={}".format(t2.shape, t2.dtype, t2.place))
            try:
                t1_cpu = t1.cpu().numpy()
                t2_cpu = t2.cpu().numpy()
                print("t1_cpu:", t1_cpu)
                print("t2_cpu:", t2_cpu)
            except Exception as e2:
                print("Failed to move to CPU:", e2)
            return False
    while not tensor_equal(ground_truth1.cpu(), shared_ipc_tensor_cpu):
        time.sleep(0.1)
    print("[DEBUG] check_ipc_tensor_xpu: ground_truth1 matched")
    event.set()
    while not tensor_equal(ground_truth2.cpu(), shared_ipc_tensor_cpu):
        time.sleep(0.1)
    print("[DEBUG] check_ipc_tensor_xpu: ground_truth2 matched")
    event.set()

# ------------------------------------------------------------------------------
# Leak checker and test base classes
# ------------------------------------------------------------------------------
class leak_checker:
    def __init__(self, test_case):
        self.checked_pids = [os.getpid()]
        self.test_case = test_case
    def __enter__(self):
        self.next_fds = self._get_next_fds(10)
        return self
    def __exit__(self, *args):
        if args[0] is None:
            self.test_case.assertFalse(self.has_shm_files())
        return False
    def check_pid(self, pid):
        self.checked_pids.append(pid)
    def _get_next_fds(self, n=1):
        fds = [os.dup(0) for i in range(n)]
        for fd in fds:
            os.close(fd)
        return fds
    def has_shm_files(self, wait=True):
        if not HAS_SHM_FILES:
            return False
        result = self._has_shm_files()
        if result and wait:
            time.sleep(0.5)
            return self._has_shm_files()
        return result
    def _has_shm_files(self):
        gc.collect()
        names = ['paddle_' + str(pid) for pid in self.checked_pids]
        for filename in os.listdir('/dev/shm'):
            for name in names:
                if filename.startswith(name):
                    print("Leak checker: found", filename)
                    return True
        return False

class TestMultiprocessingBase(unittest.TestCase):
    def get_tensor(self, device="cpu"):
        self.device = device.lower()
        tensor = paddle.zeros([5, 5], dtype="float32")
        if self.device == "xpu":
            tensor = tensor.to("xpu")
        try:
            print("Debug: get_tensor (device=%s) value: %s" % (self.device, tensor.numpy()))
            print("Debug: get_tensor id:", id(tensor))
        except Exception as e:
            print("Debug: get_tensor error:", e)
        return tensor

    def get_parameter(self, device="cpu"):
        w = paddle.nn.Layer().create_parameter(
            [10, 10],
            default_initializer=paddle.nn.initializer.Constant(value=0.0),
        )
        if self.device == "xpu":
            w = w.to("xpu")
        try:
            print("Debug: get_parameter value:", w.numpy())
            print("Debug: get_parameter id:", id(w))
        except Exception as e:
            print("Debug: get_parameter error:", e)
        return w

    def _test_empty(self, dtype="float32"):
        q = mp.Queue()
        empty = paddle.to_tensor([], dtype=dtype)
        q.put(empty)
        out = q.get(timeout=1)
        self.assertEqual(str(out), str(empty))

    def _test_sharing(self, ctx=mp, device='cpu', dtype="float32", repeat=1, param=False):
        def test_fill():
            if device.lower() == "cpu":
                x = self.get_tensor(device)
                data = [x]
            else:
                if param:
                    x = self.get_parameter(device)
                    y = x[:, 1].detach()
                else:
                    x = self.get_tensor(device)
                    y = x[:, 1].detach()
                x_ipc = maybe_share_xpu(x)
                print("[DEBUG] test_fill: parent's x IPC meta:", x_ipc)
                try:
                    x = paddle.to_tensor(paddle.base.core.LoDTensor._new_shared_xpu(x_ipc))
                except Exception as e:
                    print("[DEBUG] test_fill: Exception in new_shared_xpu for x:", e)
                    raise
                y_ipc = maybe_share_xpu(y)
                y = paddle.to_tensor(paddle.base.core.LoDTensor._new_shared_xpu(y_ipc))
                data = [x_ipc, y_ipc]
            try:
                print("Parent (before): x =", x.numpy())
                if device.lower() != "cpu":
                    print("Parent (before): y =", y.numpy())
                print("Parent tensor ids: x id =", id(x), end=", ")
                if device.lower() != "cpu":
                    print("y id =", id(y))
                else:
                    print()
            except Exception as e:
                print("Parent: Unable to print tensor values before sending:", e)
            queue = ctx.Queue()
            event = ctx.Event()
            queue.put(data)
            process = ctx.Process(target=fill_tensor, args=(queue, event))
            process.daemon = True
            lc.check_pid(process.pid)
            process.start()
            event.wait(30)
            if device.lower() == "cpu":
                y_new = x[:, 1]
                try:
                    print("Parent (after re-slice): y_new =", y_new.numpy())
                except Exception as e:
                    print("Parent: Unable to print re-sliced y:", e)
                self.assertTrue(y_new.equal(5).all(), "Re-sliced y tensor not all 5s!")
            else:
                try:
                    print("Parent (after): x =", x.numpy())
                    print("Parent (after): y =", y.numpy())
                except Exception as e:
                    print("Parent: Unable to print tensor values after process:", e)
                self.assertTrue(x.equal(5).all(), "x tensor not all 5s!")
                self.assertTrue(y.equal(5).all(), "y tensor not all 5s!")
            process.join(1 if device.lower() != "xpu" else 10)
            self.assertFalse(process.is_alive())

        def test_receive():
            queue = ctx.Queue()
            event = ctx.Event()
            process = ctx.Process(
                target=send_parambase if param else send_tensor,
                args=(queue, event, device, dtype),
            )
            process.daemon = True
            lc.check_pid(process.pid)
            process.start()
            t1 = queue.get()
            t2 = queue.get()
            if device.lower() == "xpu":
                print("[DEBUG] test_receive: before maybe_unshare_xpu, t1 type:",
                      type(t1), "value:", t1)
                t1 = maybe_unshare_xpu(t1)
                t2 = maybe_unshare_xpu(t2)
                print("[DEBUG] test_receive: after maybe_unshare_xpu, t1 shape:",
                      t1.shape, "place:", t1.place)
            try:
                print("Parent (receive): t1 =", t1.numpy())
            except Exception as e:
                print("Parent: Unable to print received t1:", e)
            self.assertTrue(t1.equal(1).all(), "Received tensor not all ones!")
            del t1, t2
            event.set()
            process.join(1 if device.lower() != "xpu" else 10)
            self.assertFalse(process.is_alive())

        with leak_checker(self) as lc:
            for _ in range(repeat):
                test_fill()
                test_receive()

class TestMultiprocessingXpu(TestMultiprocessingBase):
    @unittest.skipIf(
        not paddle.base.core.is_compiled_with_xpu(),
        "core is not compiled with XPU",
    )
    def func_test_pass_tensor(self):
        paddle.set_device("xpu")
        self._test_sharing(mp.get_context("spawn"), "xpu")
    def test_pass_tensor(self):
        self.func_test_pass_tensor()
    def test_ipc_tensor(self):
        paddle.device.set_device("xpu")
        initial_tensor = paddle.to_tensor([1, 2, 3]).to("xpu")
        bonus = paddle.to_tensor([2]).to("xpu")
        ipc_metas = initial_tensor.value().get_tensor()._share_xpu()
        print("[DEBUG] test_ipc_tensor: ipc_metas from initial_tensor:", ipc_metas)
        ctx = mp.get_context("spawn")
        event = ctx.Event()
        process = ctx.Process(target=check_ipc_tensor_xpu, args=(event, ipc_metas))
        process.daemon = True
        process.start()
        event.wait(5)
        self.assertTrue(event.is_set())
        event.clear()
        initial_tensor.add_(bonus)
        print("[DEBUG] test_ipc_tensor: after initial_tensor.add_, new value:",
              initial_tensor.cpu().numpy())
        event.wait(5)
        self.assertTrue(event.is_set())
        process.join(10)
        self.assertFalse(process.is_alive())

if __name__ == "__main__":
    unittest.main()
