// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(BooleanMaskTest, BooleanMask_test) {
  OpTester test1("BooleanMask", 1, onnxruntime::kMSDomain);
  test1.AddInput<int32_t>("data", {3,2}, {1, 2, 3, 4, 5, 6});
  test1.AddInput<bool>("mask", {3}, {true, false, true});
  test1.AddOutput<int32_t>("output", {2, 2}, {1, 2, 5, 6});
  test1.Run();

  OpTester test2("BooleanMask", 1, onnxruntime::kMSDomain);
  test2.AddInput<double>("data", {3,3,3}, {0.0915,  0.0928,  1.4071, -0.2340,  0.4966, -2.0429, -1.8860, -2.034,   1.4967,\
  -0.9386, -0.1200, -1.4857, -0.7398, -0.8596, -0.8107, 1.2828,  2.0076, -1.2885, \
  -0.4926, -0.5057,  0.3172, -0.2471, -0.1310, -0.8718, -0.4310,  0.5206,  1.5439});
  test2.AddInput<bool>("mask", {3,3}, {true, false, true, false, false, true, false, false, true});
  test2.AddOutput<double>("output", {4, 3}, {0.0915,  0.0928,  -1.8860, -2.034,   1.4967,\
  1.2828,  2.0076, -1.2885, -0.4310,  0.5206,  1.5439});
  test2.Run();
}

}  // namespace test
}  // namespace onnxruntime
