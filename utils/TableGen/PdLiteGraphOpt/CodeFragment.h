//
// Created by 骆荟州 on 2022/4/5.
//

#ifndef LLVM_CODE_FRAGMENT_H
#define LLVM_CODE_FRAGMENT_H

#include <string>

namespace PdGraphOpt {

static std::string license = R"(// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
)";

static std::string singleFileInclude = R"(#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pass.h"

)";

static std::string nameSpaceFusionBegin = R"(namespace paddle {
namespace lite {
namespace mir {
namespace fusion {
)";

static std::string nameSpaceFusionEnd = R"(}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
)";

static std::string nameSpaceMirBegin = R"(namespace paddle {
namespace lite {
namespace mir {
)";

static std::string nameSpaceMirEnd = R"(}  // namespace mir
}  // namespace lite
}  // namespace paddle
)";

static std::string rankChecker = R"(
bool assertRankEquals(const Node* node, const std::string& slotName, unsigned val) {
  auto op = const_cast<Node*>(node)->stmt()->op();
  auto scope = op->scope();
  auto &dim = scope->FindVar(op->op_info()->Input(slotName).front())
                 ->Get<lite::Tensor>().dims();
  auto rank = dim.size();
  return rank == val;
}

bool assertRankInRange(const Node* node, const std::string& slotName, unsigned low, unsigned high) {
  auto op = const_cast<Node*>(node)->stmt()->op();
  auto scope = op->scope();
  auto &dim = scope->FindVar(op->op_info()->Input(slotName).front())
                  ->Get<lite::Tensor>().dims();
  auto rank = dim.size();
  return rank >= low && rank <= high;
}
)";

static std::string genUniqueName = R"(
std::string genUniqueNameInScope(const std::string& rawName, Node* leadingNode) {
  static std::unordered_map<std::string, int> allocMap;
  int count = allocMap[rawName]++;

  char buf[20];
  sprintf(buf, "%16llx", (long long)leadingNode);
  std::string addrStr(buf + 8);

  return rawName + "_" + to_string(count) + "_" + addrStr;
}
)";

static std::string directElementWiseCompute = R"(
TensorLite directEleWiseAddFloat32(TensorLite *a, float b) {
  TensorLite res;
  res.CopyDataFrom(*a);
  auto *resData = res.mutable_data<float>();
  unsigned long size = a->data_size();
  for (unsigned long i = 0; i < size; i++) {
    resData[i] += b;
  }
  return res;
}

TensorLite directEleWiseAddFloat32(float a, TensorLite *b) {
  return directEleWiseAddFloat32(b, a);
}

TensorLite directEleWiseAddFloat32(TensorLite *a, TensorLite *b) {
  TensorLite res;

  if (a->dims() == b->dims()) {
    unsigned long size = a->data_size();
    res.CopyDataFrom(*a);
    auto *resData = res.mutable_data<float>();
    const auto *bData = b->data<float>();

    for (unsigned long i = 0; i < size; i++) {
      resData[i] += bData[i];
    }
  }
  else if (a->data_size() == 1 || b->data_size() == 1) {
    float broadcastValue = a->data_size() == 1 ?
          *(a->data<float>()) : *(b->data<float>());
    return directEleWiseAddFloat32(a->data_size() == 1 ? b : a,
                                   broadcastValue);
  }
  else {
    LOG_FATAL;
  }
  return res;
}

TensorLite directEleWiseSubFloat32(TensorLite *a, float b) {
  TensorLite res;
  res.CopyDataFrom(*a);
  auto *resData = res.mutable_data<float>();
  unsigned long size = a->data_size();
  for (unsigned long i = 0; i < size; i++) {
    resData[i] -= b;
  }

  return res;
}

TensorLite directEleWiseSubFloat32(float a, TensorLite *b) {
  TensorLite res;
  res.CopyDataFrom(*b);
  auto *resData = res.mutable_data<float>();
  unsigned long size = b->data_size();
  for (unsigned long i = 0; i < size; i++) {
    resData[i] = a - resData[i];
  }

  return res;
}

TensorLite directEleWiseSubFloat32(TensorLite *a, TensorLite *b) {
  TensorLite res;

  if (a->dims() == b->dims()) {
    unsigned long size = a->data_size();
    res.CopyDataFrom(*a);
    auto *resData = res.mutable_data<float>();
    const auto *bData = b->data<float>();

    for (unsigned long i = 0; i < size; i++) {
      resData[i] -= bData[i];
    }
  }
  else if (a->data_size() == 1 || b->data_size() == 1) {
    if (a->data_size() == 1) {
      return directEleWiseSubFloat32(*(a->data<float>()), b);
    }
    else {
      return directEleWiseSubFloat32(a, *(b->data<float>()));
    }
  }
  else {
    LOG_FATAL;
  }
  return res;
}

TensorLite directEleWiseMulFloat32(TensorLite *a, float b) {
  TensorLite res;
  res.CopyDataFrom(*a);
  auto *resData = res.mutable_data<float>();
  unsigned long size = a->data_size();
  for (unsigned long i = 0; i < size; i++) {
    resData[i] *= b;
  }

  return res;
}

TensorLite directEleWiseMulFloat32(float a, TensorLite *b) {
    return directEleWiseMulFloat32(b, a);
}

TensorLite directEleWiseMulFloat32(TensorLite *a, TensorLite *b) {
  TensorLite res;

  if (a->dims() == b->dims()) {
    unsigned long size = a->data_size();
    res.CopyDataFrom(*a);
    auto *resData = res.mutable_data<float>();
    const auto *bData = b->data<float>();

    for (unsigned long i = 0; i < size; i++) {
      resData[i] *= bData[i];
    }
  }
  else if (a->data_size() == 1 || b->data_size() == 1) {
    float broadcastValue = a->data_size() == 1 ?
                           *(a->data<float>()) : *(b->data<float>());
    return directEleWiseMulFloat32(a->data_size() == 1 ? b : a,
                                   broadcastValue);
  }
  else {
    LOG_FATAL;
  }
  return res;
}

TensorLite directEleWiseDivFloat32(TensorLite *a, float b) {
  TensorLite res;
  res.CopyDataFrom(*a);
  auto *resData = res.mutable_data<float>();
  unsigned long size = a->data_size();
  for (unsigned long i = 0; i < size; i++) {
    resData[i] /= b;
  }
  return res;
}

TensorLite directEleWiseDivFloat32(float a, TensorLite *b) {
  TensorLite res;
  res.CopyDataFrom(*b);
  auto *resData = res.mutable_data<float>();
  unsigned long size = b->data_size();
  for (unsigned long i = 0; i < size; i++) {
    resData[i] = a / resData[i];
  }
  return res;
}

TensorLite directEleWiseDivFloat32(TensorLite *a, TensorLite *b) {
  TensorLite res;

  if (a->dims() == b->dims()) {
    unsigned long size = a->data_size();
    res.CopyDataFrom(*a);
    auto *resData = res.mutable_data<float>();
    const auto *bData = b->data<float>();

    for (unsigned long i = 0; i < size; i++) {
      resData[i] /= bData[i];
    }
  }
  else if (a->data_size() == 1 || b->data_size() == 1) {
    if (a->data_size() == 1) {
      return directEleWiseDivFloat32(*(a->data<float>()), b);
    }
    else {
      return directEleWiseDivFloat32(a, *(b->data<float>()));
    }
  }
  else {
    LOG_FATAL;
  }
  return res;
}

/**
 * 将a压平为二维矩阵，对矩阵每一行应用一个缩放因子
 * 这个缩放因子就是b中具体的一个值（标量）
 * 因此要求a的行数等于b的数据总数。
 */
TensorLite directRowWiseMulFloat32(TensorLite *a,
                                   TensorLite *b,
                                   unsigned a_num_col_dims) {

  auto aDim = a->dims();
  int aDimSize = aDim.size();
  if (a_num_col_dims > aDimSize || a_num_col_dims < 0) {
    LOG_FATAL;
  }
  unsigned colCount = 1;
  if (a_num_col_dims > 0) {
    for (int i = 0; i < a_num_col_dims; i++) {
      colCount *= aDim[aDimSize - 1 - i];
    }
  }
  unsigned rowCount = a->data_size() / colCount;
  if (b->data_size() != rowCount) {
    LOG_FATAL;
  }
  TensorLite res;
  res.CopyDataFrom(*a);
  auto *resData = res.mutable_data<float>();
  const auto *bData = b->data<float>();

  for (unsigned i = 0; i < rowCount; i++) {
    for (unsigned j = 0; j < colCount; j++) {
      resData[i * colCount + j] *= bData[i];
    }
  }

  return res;
}

TensorLite directEleWiseSqrtFloat32(TensorLite *a) {
  TensorLite res;
  res.CopyDataFrom(*a);
  float* data = res.mutable_data<float>();
  for (unsigned i = 0; i < res.data_size(); i++) {
    data[i] = std::sqrt(data[i]);
  }
  return res;
}
)";

} // end namespace PdGraphOpt


#endif // LLVM_CODE_FRAGMENT_H
