// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/boolean_mask.h"

namespace onnxruntime {
namespace contrib     {

ONNX_OPERATOR_KERNEL_EX(
    BooleanMask,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T",    DataTypeImpl::AllTensorTypes()),
    BooleanMask);

Status BooleanMaskBase::PrepareForCompute(OpKernelContext* context, Prepare& p) const {

  auto input_tensor  = context->Input<Tensor>(0);
  auto mask_tensor = context->Input<Tensor>(1);
  ORT_ENFORCE(input_tensor  != nullptr);
  ORT_ENFORCE(mask_tensor != nullptr);

  auto input_shape   = input_tensor->Shape();
  auto mask_shape  = mask_tensor->Shape();
  auto input_rank = input_shape.NumDimensions();
  auto mask_rank = mask_shape.NumDimensions();

  if (mask_rank == 0 || input_rank == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
      "input tensor and mask tensor must have rank larger than 0");
  }
  
  if (mask_rank > input_rank) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
      "input tensor rank should not be smaller than mask tensor rank");
  }

  for (auto i = 0; i < mask_rank - 1; ++i) {
      if (input_shape[i] != mask_shape[i]) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
        "input tensor shape does not match mask tensor shape, i=", i);
      }
  }

  p.element_bytes = input_tensor->DataType()->Size();
  p.element_to_copy = 1;
  for (auto i = mask_rank; i < input_rank; ++i) {
    p.element_to_copy *= static_cast<uint64_t>(input_shape[i]);
  }
  p.bytes_to_copy = p.element_bytes * p.element_to_copy;
  
  uint64_t mask_size = 1;
  for (auto i = 0; i < mask_rank; ++i) {
    mask_size *= static_cast<uint64_t>(mask_shape[i]);
  }
  
  auto mask_number = 0;
  for (auto i = 0; i < mask_size; ++i) {
    auto mask_value = reinterpret_cast<const bool*>(static_cast<const char*>(mask_tensor->DataRaw()) + i);
    if (*mask_value)
    {
        p.element_offsets.emplace_back(i);
        ++ mask_number;
    }
  } 

  std::vector<int64_t> shape(mask_shape.GetDims().begin(),
                             mask_shape.GetDims().end());
  shape.insert(shape.begin(), mask_number);
  auto output_tensor = context->Output(0,TensorShape(shape));

  if (input_tensor->DataType() == DataTypeImpl::GetType<std::string>()) {
    p.input_str_base  = static_cast<const std::string*>(input_tensor->DataRaw());
    p.output_str_base = static_cast<std::string*>(output_tensor->MutableDataRaw());
  } else {
    p.input_base      = static_cast<const uint8_t*>(input_tensor->DataRaw());
    p.output_base     = static_cast<uint8_t*>(output_tensor->MutableDataRaw());
  }

  return Status::OK();
}

Status BooleanMask::Compute(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(context, p));
  return nullptr == p.input_str_base ? GatherNumber(p) : GatherString(p);
}

Status BooleanMask::GatherNumber(const Prepare& p) const {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < static_cast<int64_t>(p.element_offsets.size()); ++i) {
    memcpy(p.output_base + i * p.bytes_to_copy,
           p.input_base + p.element_offsets[i] * p.element_bytes,
           p.bytes_to_copy);
  }
  return Status::OK();
}

Status BooleanMask::GatherString(const Prepare& p) const {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < static_cast<int64_t>(p.element_offsets.size()); ++i) {
    for (int64_t j = 0; j < static_cast<int64_t>(p.element_to_copy); ++j) {
      p.output_str_base[i * p.element_to_copy + j] = p.input_str_base[p.element_offsets[i] + j];
    }
  }
  return Status::OK();
}

}
}
