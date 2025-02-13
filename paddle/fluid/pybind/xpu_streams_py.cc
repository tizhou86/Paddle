// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/pybind/xpu_streams_py.h"

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"

#include "paddle/phi/api/profiler/event.h"
#include "paddle/phi/core/platform/device_event_base.h"

namespace py = pybind11;

namespace paddle {
namespace platform {
#ifdef PADDLE_WITH_XPU
XPUStream get_current_stream(int device_id) {
  if (device_id == -1) {
    device_id = phi::backends::xpu::GetXPUCurrentDeviceId();
  }
  auto place = phi::XPUPlace(device_id);
  auto *dev_ctx = static_cast<phi::XPUContext *>(
      phi::DeviceContextPool::Instance().Get(place));
  dev_ctx->Wait();
  return dev_ctx->stream();
}

// XPUStream set_current_stream(XPUStream stream) {
//   auto *original_stream = get_current_stream(stream->place().GetDeviceId());
//   auto *xpu_context = static_cast<phi::XPUContext *>(
//       DeviceContextPool::Instance().Get(stream->place()));
//   xpu_context->SetStream(stream, /*clear=*/false);
//   return original_stream;
// }

#endif
}  // namespace platform
namespace pybind {
void BindXpuStream(py::module *m_ptr) {
  auto &m = *m_ptr;

  // Bind Methods
  m.def("_xpu_device_synchronize", [](int device_id) {
#ifdef PADDLE_WITH_XPU
    if (device_id == -1) {
      device_id = paddle::platform::GetXPUCurrentDeviceId();
    }
    int curr_device_id = paddle::platform::GetXPUCurrentDeviceId();
    paddle::platform::SetXPUDeviceId(device_id);
    auto place = phi::XPUPlace(device_id);
    auto *dev_ctx = static_cast<phi::XPUContext *>(
        phi::DeviceContextPool::Instance().Get(place));
    dev_ctx->Wait();
    paddle::platform::SetXPUDeviceId(curr_device_id);
#else
    PADDLE_THROW(common::errors::Unavailable(
        "Paddle is not compiled with XPU. Cannot visit device synchronize."));
#endif
  });

  m.def(
      "_get_current_stream",
      [](int device_id) {
#ifdef PADDLE_WITH_XPU
        if (device_id == -1) {
          device_id = paddle::platform::GetXPUCurrentDeviceId();
        }
        paddle::platform::SetXPUDeviceId(device_id);
        return platform::get_current_stream(device_id);
#else
        PADDLE_THROW(
            common::errors::Unavailable("Paddle is not compiled with CUDA. "
                                        "Cannot visit device synchronize."));
#endif
      },
      py::return_value_policy::reference);

//   m.def(
//       "_set_current_stream",
//       [](XPUStream stream) {
// #ifdef PADDLE_WITH_XPU
//         return platform::set_current_stream(stream);
// #else
//         PADDLE_THROW(
//             common::errors::Unavailable("Paddle is not compiled with CUDA. "
//                                         "Cannot visit device synchronize."));
// #endif
//       },
//       py::return_value_policy::reference);

  m.def("_device_synchronize", [](int device_id) {
#ifdef PADDLE_WITH_XPU
    if (device_id == -1) {
      device_id = paddle::platform::GetXPUCurrentDeviceId();
    }

    int curr_device_id = paddle::platform::GetXPUCurrentDeviceId();
    paddle::platform::SetXPUDeviceId(device_id);
    PADDLE_ENFORCE_XPU_SUCCESS(cudaDeviceSynchronize());
    paddle::platform::SetXPUDeviceId(curr_device_id);
#else
    PADDLE_THROW(common::errors::Unavailable(
        "Paddle is not compiled with CUDA. Cannot visit device synchronize."));
#endif
  });

  py::class_<XPUStream>(m, "XPUStream", R"DOC(
      The handle of the CUDA stream.

      Parameters:
          device(paddle.CUDAPlace()|int|None, optional): The device which wanted to allocate the stream.
              If device is None or negative integer, device will be the current device.
              If device is positive integer, it must less than the device count. Default: None.
          priority(int|None, optional): The priority of stream. The priority can be 1(high) or 2(normal).
              If priority is None, the priority is 2(normal). Default: None.

      Examples:
          .. code-block:: python

              >>> # doctest: +REQUIRES(env:GPU)
              >>> import paddle
              >>> s1 = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
              >>> s2 = paddle.device.cuda.Stream(0, 1)
              >>> s3 = paddle.device.cuda.Stream()

      )DOC")
// #ifdef PADDLE_WITH_XPU
      // .def(
      //     "wait_event",
      //     [](phi::XPUCUDAStream &self, phi::CudaEvent &event) {
      //       self.WaitEvent(event.GetRawCudaEvent());
      //     },
      //     R"DOC(
      //     Makes all future work submitted to stream wait for all work captured in event.

      //     Parameters:
      //         event(CUDAEvent): The event to wait on.

      //     Examples:
      //         .. code-block:: python

      //             >>> # doctest: +REQUIRES(env:GPU)
      //             >>> import paddle
      //             >>> s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
      //             >>> event = paddle.device.cuda.Event()
      //             >>> s.wait_event(event)
      //     )DOC")
      // .def(
      //     "wait_stream",
      //     [](phi::XPUCUDAStream &self, phi::XPUCUDAStream &stream) {
      //       phi::CudaEvent event;
      //       event.Record(stream.raw_stream());
      //       self.WaitEvent(event.GetRawCudaEvent());
      //     },
      //     R"DOC(
      //     Synchronizes with the given stream.

      //     Parameters:
      //         stream(XPUCUDAStream): The stream to synchronize with.

      //     Examples:
      //         .. code-block:: python

      //             >>> # doctest: +REQUIRES(env:GPU)
      //             >>> import paddle
      //             >>> s1 = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
      //             >>> s2 = paddle.device.cuda.Stream(0, 1)
      //             >>> s1.wait_stream(s2)

      //     )DOC")
      // .def(
      //     "query",
      //     [](phi::XPUCUDAStream &self) { return self.Query(); },
      //     R"DOC(
      //     Return the status whether if all operations in stream have completed.

      //     Returns: A boolean value.

      //     Examples:
      //         .. code-block:: python

      //             >>> # doctest: +REQUIRES(env:GPU)
      //             >>> import paddle
      //             >>> s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
      //             >>> is_done = s.query()

      //     )DOC")
      .def(
          "synchronize",
          [](XPUStream &self) { xpu_wait(self); },
          R"DOC(
          Waits for stream tasks to complete.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:GPU)
                  >>> import paddle
                  >>> s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
                  >>> s.synchronize()

          )DOC");
//       .def(
//           "record_event",
//           [](phi::XPUCUDAStream &self, phi::CudaEvent *event) {
//             if (event == nullptr) {
//               event = new phi::CudaEvent();
//             }
//             event->Record(self.raw_stream());
//             return event;
//           },
//           R"DOC(
//           Record a CUDA event in the stream.

//           Parameters:
//               event(CUDAEvent, optional): The event to be record. If event is None, a new event is created.
//                   Default: None.

//           Returns:
//               The record event.

//           Examples:
//               .. code-block:: python

//                   >>> # doctest: +REQUIRES(env:GPU)
//                   >>> import paddle
//                   >>> s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
//                   >>> event = s.record_event()

//           )DOC",
//           py::arg("event") = nullptr)
//       .def_property_readonly(
//           "cuda_stream",
//           [](phi::XPUCUDAStream &self) {
//             VLOG(10) << self.raw_stream();
//             return reinterpret_cast<std::uintptr_t>(self.raw_stream());
//           },
//           R"DOC(
//           return the raw cuda stream of type cudaStream_t as type int.

//           Examples:
//               .. code-block:: python

//                   >>> # doctest: +REQUIRES(env:GPU)
//                   >>> import paddle
//                   >>> import ctypes
//                   >>> cuda_stream = paddle.device.cuda.current_stream().cuda_stream
//                   >>> print(cuda_stream)

//                   >>> ptr = ctypes.c_void_p(cuda_stream)  # convert back to void*
//                   >>> print(ptr)

//           )DOC")
//       .def_property_readonly(
//           "place",
//           [](phi::XPUCUDAStream &self) { return phi::XPUPlace(self.place()); })
// #endif
//       .def(
//           "__init__",
//           [](phi::XPUCUDAStream &self, phi::XPUPlace *place, int priority) {
// #ifdef PADDLE_WITH_XPU
//             if (priority != 1 && priority != 2) {
//               PADDLE_THROW(common::errors::InvalidArgument(
//                   "Priority should be 1(high) or 2(normal) "));
//             }

//             auto stream_flag = phi::XPUCUDAStream::StreamFlag::kStreamNonBlocking;
//             if (place == nullptr) {
//               int curr_device_id = platform::GetXPUCurrentDeviceId();
//               auto place_tmp = phi::XPUPlace(curr_device_id);
//               new (&self) phi::XPUCUDAStream(place_tmp, priority - 2, stream_flag);
//             } else {
//               // setting priority 1(high) and 2(normal) correspond to the actual
//               // cuda stream priority -1 and 0.
//               new (&self) phi::XPUCUDAStream(*place, priority - 2, stream_flag);
//             }
// #else
//             PADDLE_THROW(common::errors::Unavailable(
//         "Class XPUCUDAStream can only be initialized on the GPU platform."));
// #endif
//           },
//           py::arg("device") = nullptr,
//           py::arg("priority") = 2)
//       .def(
//           "__init__",
//           [](phi::XPUCUDAStream &self, int device, int priority) {
// #ifdef PADDLE_WITH_XPU
//             if (priority != 1 && priority != 2) {
//               PADDLE_THROW(common::errors::InvalidArgument(
//                   "Priority should be 1(high) or 2(normal) "));
//             }

//             int device_count = platform::GetDeviceCount();
//             if (device < 0) {
//               device = platform::GetXPUCurrentDeviceId();
//             }
//             if (device >= device_count) {
//               PADDLE_THROW(common::errors::InvalidArgument(
//                   "The device id  must be inside [0, %d), but input device=%d.",
//                   device_count,
//                   device));
//             }

//             auto stream_flag = phi::XPUCUDAStream::StreamFlag::kStreamNonBlocking;
//             // setting priority 1(high) and 2(normal) correspond to the actual
//             // cuda stream priority -1 and 0.
//             new (&self) phi::XPUCUDAStream(
//                 phi::XPUPlace(device), priority - 2, stream_flag);
// #else
//             PADDLE_THROW(common::errors::Unavailable(
//         "Class XPUCUDAStream can only be initialized on the GPU platform."));
// #endif
//           },
//           py::arg("device") = -1,
//           py::arg("priority") = 2)
//       .def("__init__", [](phi::XPUCUDAStream &self) {
// #ifdef PADDLE_WITH_XPU
//         int device_id = platform::GetXPUCurrentDeviceId();
//         auto stream_flag = phi::XPUCUDAStream::StreamFlag::kStreamNonBlocking;
//         new (&self) phi::XPUCUDAStream(
//             phi::XPUPlace(device_id), /*priority=*/0, stream_flag);
// #else
//             PADDLE_THROW(common::errors::Unavailable(
//         "Class XPUCUDAStream can only be initialized on the GPU platform."));
// #endif
//       });

//   py::class_<phi::CudaEvent>(m, "CUDAEvent", R"DOC(
//       The handle of the CUDA event.

//       Parameters:
//           enable_timing(bool, optional): Whether the event will measure time. Default: False.
//           blocking(bool, optional): Whether the wait() func will be blocking. Default: False;
//           interprocess(bool, optional): Whether the event can be shared between processes. Default: False.

//       Examples:
//           .. code-block:: python

//               >>> # doctest: +REQUIRES(env:GPU)
//               >>> import paddle
//               >>> event = paddle.device.cuda.Event()

//       )DOC")
// #ifdef PADDLE_WITH_XPU
//       .def(
//           "record",
//           [](phi::CudaEvent &self, phi::XPUCUDAStream *stream) {
//             if (stream == nullptr) {
//               stream = paddle::platform::get_current_stream(-1);
//             }
//             self.Record(stream->raw_stream());
//           },
//           R"DOC(
//           Records the event in the given stream.

//           Parameters:
//               stream(XPUCUDAStream, optional): The handle of CUDA stream. If None, the stream is the current stream. Default: None.

//           Examples:
//               .. code-block:: python

//                   >>> # doctest: +REQUIRES(env:GPU)
//                   >>> import paddle
//                   >>> paddle.device.set_device('gpu')
//                   >>> event = paddle.device.cuda.Event()
//                   >>> event.record()

//           )DOC",
//           py::arg("stream") = nullptr)
//       .def(
//           "query",
//           [](phi::CudaEvent &self) { return self.Query(); },
//           R"DOC(
//           Queries the event's status.

//           Returns: A boolean which indicates all work currently captured by the event has been completed.

//           Examples:
//               .. code-block:: python

//                   >>> # doctest: +REQUIRES(env:GPU)
//                   >>> import paddle
//                   >>> paddle.device.set_device('gpu')
//                   >>> event = paddle.device.cuda.Event()
//                   >>> is_done = event.query()

//           )DOC")
//       .def(
//           "elapsed_time",
//           [](phi::CudaEvent &self, phi::CudaEvent &end_event) {
//             return self.ElapsedTime(&end_event);
//           },
//           R"DOC(
//           Returns the time elapsed in milliseconds after the event was
//           recorded and before the end_event was recorded.

//           Returns: A int which indicates the elapsed time.

//           Examples:
//               .. code-block:: python

//                   >>> # doctest: +REQUIRES(env:GPU)
//                   >>> import paddle

//                   >>> paddle.set_device('gpu')
//                   >>> e1 = paddle.device.Event(enable_timing=True)
//                   >>> e1.record()

//                   >>> e2 = paddle.device.Event(enable_timing=True)
//                   >>> e2.record()
//                   >>> e1.elapsed_time(e2)

//           )DOC")
//       .def(
//           "synchronize",
//           [](phi::CudaEvent &self) { self.Synchronize(); },
//           R"DOC(
//             Waits for an event to complete.

//             Examples:
//                 .. code-block:: python

//                     >>> # doctest: +REQUIRES(env:GPU)
//                     >>> import paddle
//                     >>> paddle.device.set_device('gpu')
//                     >>> event = paddle.device.cuda.Event()
//                     >>> event.synchronize()

//           )DOC")
// #endif
//       .def(
//           "__init__",
//           [](phi::CudaEvent &self,
//              bool enable_timing,
//              bool blocking,
//              bool interprocess) {
// #ifdef PADDLE_WITH_XPU
//             unsigned int flags = platform::GenerateDeviceEventFlag(
//                 enable_timing, blocking, interprocess);
//             new (&self) phi::CudaEvent(flags);
// #else
//             PADDLE_THROW(common::errors::Unavailable(
//                 "Class CUDAEvent can only be initialized on the GPU "
//                 "platform."));

// #endif
//           },
//           py::arg("enable_timing") = false,
//           py::arg("blocking") = false,
//           py::arg("interprocess") = false);
}

}  // namespace pybind
}  // namespace paddle
