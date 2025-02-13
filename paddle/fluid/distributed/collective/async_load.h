/****************************************************
 * async_load.h (XPU, skipping DeviceEvent usage)
 ****************************************************/
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

//#include "paddle/fluid/platform/device_context.h"
//#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/platform/device_event_base.h"
//#include "paddle/phi/core/places.h"

namespace paddle {
namespace distributed {

using Place = phi::Place;

/**
 * AsyncLoad that does NOT use platform::DeviceEvent if place == XPU.
 */
class AsyncLoad {
 public:
  class Task {
   public:
    explicit Task(const Place& place);
    virtual ~Task();

    bool IsCompleted();

    // Replaces CudaSynchronize with XpuSynchronize
    void XpuSynchronize();
    void CpuSynchronize();

    // If not XPU, record the event. If XPU, do nothing
    void UpdateWaitChain(const phi::DeviceContext& ctx);

   private:
    bool use_event_;  // false if place is XPU
    platform::DeviceEvent load_event_;
    Place task_place_;
  };

  // Offload
  std::shared_ptr<Task> Offload(phi::DenseTensor* dst, const phi::DenseTensor& src);

  // OffloadWithOffset
  std::shared_ptr<Task> OffloadWithOffset(phi::DenseTensor* dst,
                                          const phi::DenseTensor& src,
                                          size_t dst_offset,
                                          size_t src_offset,
                                          size_t offload_size);

  // Reload
  std::shared_ptr<Task> Reload(phi::DenseTensor* dst, const phi::DenseTensor& src);

 private:
  bool is_initialized_{false};

  // A fallback "offload context," though we won't do multi-stream sync for XPU
  std::unique_ptr<phi::XPUContext> load_ctx_;
  Place xpu_place_;

  std::shared_ptr<Task> CreateTask(const Place& place);

  // If not XPU, store calc-event. If XPU, skip
  std::unordered_map<std::string, platform::DeviceEvent> place_to_calc_event_;

  // Prepare env
  void PrepareLoadEnv(const std::string& key, const Place& place);

  // If not XPU, do event sync. If XPU, skip
  void SyncCalcuStream(const Place& place,
                       phi::XPUContext* offload_ctx,
                       platform::DeviceEvent& calc_event);
};

}  // namespace distributed
}  // namespace paddle
