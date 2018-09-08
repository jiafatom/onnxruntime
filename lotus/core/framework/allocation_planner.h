#pragma once

#include "core/common/status.h"
#include "core/framework/alloc_kind.h"
#include "core/framework/allocator.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/graph/graph.h"
namespace onnx {
class TensorShapeProto;
}
namespace Lotus {

class ExecutionProviders;
class KernelRegistryManager;
class MLValueNameIdxMap;

// ISequentialPlannerContext abstracts how the planner accesses information (such as inferred shape)
// to do the planning.
class ISequentialPlannerContext {
 public:
  virtual const onnx::TensorShapeProto* GetShape(const LotusIR::NodeArg& arg) const = 0;
  virtual bool EnableParallelExecution() const { return false; }
};

class SequentialPlannerContext : public ISequentialPlannerContext {
 public:
  SequentialPlannerContext()
      : m_enable_parallel_execution(false) {
  }

  SequentialPlannerContext(bool p_enable_parallel_execution)
      : m_enable_parallel_execution(p_enable_parallel_execution) {
  }

  const onnx::TensorShapeProto* GetShape(const LotusIR::NodeArg& arg) const override {
    return arg.Shape();
  }

  bool EnableParallelExecution() const override {
    return m_enable_parallel_execution;
  }

 private:
  bool m_enable_parallel_execution;
};

class SequentialPlanner {
 public:
  // This API allows user to provide a custom planner context.
  static Status CreatePlan(const LotusIR::Graph& graph,
                           const ExecutionProviders& providers,
                           const KernelRegistryManager& kernel_registry,
                           const MLValueNameIdxMap& mlvalue_name_idx_map,
                           const ISequentialPlannerContext& context,
                           std::unique_ptr<SequentialExecutionPlan>& plan);

  // This uses a standard planner context and is meant to be the primary API for creating a plan
  // as the context is primarily used in test scenarios.
  static Status CreatePlan(const LotusIR::Graph& graph,
                           const ExecutionProviders& providers,
                           const KernelRegistryManager& kernel_registry,
                           const MLValueNameIdxMap& mlvalue_name_idx_map,
                           std::unique_ptr<SequentialExecutionPlan>& plan) {
    SequentialPlannerContext context;
    return CreatePlan(graph, providers, kernel_registry, mlvalue_name_idx_map, context, plan);
  }
};

}  // namespace Lotus
