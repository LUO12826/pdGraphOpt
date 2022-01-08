//===- SkeletonEmitter.cpp - Skeleton TableGen backend          -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This Tablegen backend emits ...
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
#include <set>
#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>

#define DEBUG_TYPE "graph-opt-pass-emitter"

using namespace llvm;

namespace {

//对于每个pass生成4部分，内容分别对应原来的.h file and .cpp file for pass class and fuser class.
class GraphOptPassEmitter {
  
private:
  RecordKeeper &Records;
  
  void EmitFuserHeader(Record*, raw_ostream&);
  
  void EmitFuserImpl(Record*, raw_ostream&);
  
  void EmitBuildPatternMethod(Record*, raw_ostream&);
  
  void EmitInsertNewNodeMethod(Record*, raw_ostream&);
  
  void EmitGenOpDescMethod(Record*, raw_ostream&);
  
  void EmitPassHeader(Record*, raw_ostream&);
  
  void EmitPassImpl(Record*, raw_ostream&);
  
  StringRef dfsPatDag(DagInit* dag, raw_ostream& os);
  
  ///Pattern（有向图）的首个节点，融合时会作为替换点位
  DagInit* firstOpDag{nullptr};

public:
  GraphOptPassEmitter(RecordKeeper &RK) : Records(RK) {}

  void run(raw_ostream &OS);
  
};

} // anonymous namespace

//辅助函数
namespace {

//从一个Pat中获取resultPattern的便捷方法
DagInit* getResultPatFromPat(Record* record) {
  auto resultPatternsArr = record->getValueAsListInit("resultPatterns")->getValues();
  if (resultPatternsArr.size() < 1) {
    PrintFatalError("Pat has no result pattern.");
  }
  return dyn_cast<DagInit>(resultPatternsArr[0]);
}

//从DagInit中获取dag的operator的便捷方法
Record* getOpFromDagInit(DagInit* dag) {
  auto opInit = dyn_cast<DefInit>(dag->getOperator());
  if(opInit) {
    return opInit->getDef();
  } else {
    PrintFatalError("No op or op is not a record.");
  }
}

} // anonymous namespace

//遍历sourcePattern，生成定义VarNode、OpNode及其定义它们拓扑关系的代码
StringRef GraphOptPassEmitter::dfsPatDag(DagInit* dag, raw_ostream& os) {
  
  Record* op = getOpFromDagInit(dag);
  StringRef opType = op->getValueAsString("type");
  StringRef opKey = op->getValueAsString("key");
  ArrayRef<Init*> args = dag->getArgs();
  
  //生成该dag的op对应的OpNode
  os << "  //Start op " << opKey << "\n";
  os << llvm::formatv("  auto* {0} = OpNode(\"{1}\", \"{2}\");",
                      opKey, opKey, opType);
  os << "\n";
  os << "  " << opKey << "->AsIntermediate();";
  os << "\n\n";
  
  
  //检查这个dag里面是否还嵌套着另一个dag
  bool patWithoutDag = true;
  for(Init* init : args) {
    if (isa<DagInit>(init)) {
      patWithoutDag = false;
      break;
    }
  }
  
  //这个OpNode的input
  SmallVector<StringRef, 4> inputs;
  
  //如果这个dag没有嵌套的dag，生成它的VarNode时会生成一个assert_is_op_input调用。
  if (patWithoutDag) {
    this->firstOpDag = dag;
    auto opArgs = op->getValueAsDag("arguments");
    
    unsigned currentArg = 0;
    for(Init* init : args) {
      auto varArgRec = dyn_cast<DefInit>(init)->getDef();
      StringRef varName = varArgRec->getValueAsString("name");
      inputs.push_back(varName);
      
      os << llvm::formatv("  auto* {0} = VarNode(\"{1}\")->assert_is_op_input(\"{2}\", \"{3}\");",
                          varName, varName, opType, opArgs->getArgNameStr(currentArg++));
      os << "\n";
    }
  }
  //如果这个dag有嵌套的dag，生成它的VarNode时会生成一个assert_is_persistable_var调用。
  //目前这样的处理仅仅是根据paddle中FcFuser得出的。
  else {
    for(Init* init : args) {
      //如果这个dag的参数是DefInit，也就是说它是个Var，所以生成一个VarNode
      if (isa<DefInit>(init)) {
        auto varArgRec = dyn_cast<DefInit>(init)->getDef();
        if (varArgRec->getType()->getAsString() == "TensorVar") {
          StringRef varName = varArgRec->getValueAsString("name");
          inputs.push_back(varName);
          
          os << llvm::formatv("  auto* {0} = VarNode(\"{1}\")->assert_is_persistable_var();",
                              varName, varName);
          os << "\n";
        }
        
      }
      //如果这个dag的参数是DagInit，也就是说它是个Dag，所以递归调用本方法。
      else if(isa<DagInit>(init)) {
        StringRef innerOpKey = dfsPatDag(dyn_cast<DagInit>(init), os);
        std::string innerOpOutKey = innerOpKey.str() + "_out";
        
        inputs.push_back(StringRef(innerOpOutKey));
        
        os << llvm::formatv("  auto* {0} = VarNode(\"{1}\");",
                            innerOpOutKey, innerOpOutKey);
        os << "\n";
        os << "  " << innerOpOutKey << "->AsIntermediate();";
        os << "\n";
        
        //连接op的输入和输出。
        os << llvm::formatv("  {0}_inputs >> *{1} >> *{2};",
                            innerOpKey, innerOpKey, innerOpOutKey);
        os << "\n";
      }
    }
  }
  
  //生成该op的input集合
  std::string opInputSet = opKey.str() + "_inputs";
  os << "  std::vector<PMNode*> " << opInputSet << " {";
  bool first = true;
  for(StringRef& input : inputs) {
    if (first) {
      os << input;
      first = false;
    }
    else os << ", " << input;
  }
  os << "};\n";
  os << "  //End op " << opKey << "\n";
  return opKey;
}

//生成源码中`GenOpDesc`这个方法。
void GraphOptPassEmitter::EmitGenOpDescMethod(Record* record, raw_ostream& os) {
  if (!firstOpDag) {
    PrintFatalError("firstOpDag not found.");
  }
  
  Record* patHeadOp = getOpFromDagInit(firstOpDag);

  auto patName = record->getName().drop_back(3);
  os << "cpp::OpDesc " << patName << "Fuser::GenOpDesc(const key2nodes_t& matched) {\n";
  
  StringRef patHeadOpKey = patHeadOp->getValueAsString("key");
  os << llvm::formatv("  auto op_desc = *matched.at(\"{0}\")->stmt()->op_info();\n", patHeadOpKey);
  
  auto patHeadOpArgNames = patHeadOp->getValueAsDag("arguments")->getArgNames();
  for(auto arg : patHeadOpArgNames) {
    StringRef argName = arg->getValue();
    os << llvm::formatv("  auto input_{0}_name = op_desc.Input(\"{1}\").front();\n", argName, argName);
    os << llvm::formatv("  std::vector<float> {0}_scale_vct;\n", argName);
  }
  
  if (patHeadOpArgNames.size() == 1) {
    os << llvm::formatv("  bool is_quantized_op = op_desc.HasInputScale(input_{0}_name);\n",
                        patHeadOpArgNames.front()->getValue());
  }
  else {
    for(size_t i = 0, argCount = patHeadOpArgNames.size(); i < argCount - 1; i++) {
      StringRef argName = patHeadOpArgNames[i]->getValue();
      if (i == 0) {
        os << llvm::formatv("  bool is_quantized_op = op_desc.HasInputScale(input_{0}_name) &&\n", argName);
      }
      else {
        os << llvm::formatv("                         op_desc.HasInputScale(input_{0}_name) &&\n", argName);
      }
    }
    os << llvm::formatv("                         op_desc.HasInputScale(input_{0}_name);\n", patHeadOpArgNames.back()->getValue());
  }
  
  os << "  if (is_quantized_op) {\n";
  for(auto argNameInit : patHeadOpArgNames) {
    StringRef argName = argNameInit->getValue();
    os << llvm::formatv("    {0}_scale_vct = op_desc.GetInputScale(input_{1}_name);\n", argName, argName);
  }
  os << "  }\n";
  
  os << "  op_desc.mutable_inputs()->clear();\n";
  os << "  op_desc.mutable_outputs()->clear();\n";
  
  DagInit* resultPattern = getResultPatFromPat(record);
  Record* resultPatOp = getOpFromDagInit(resultPattern);
  
  os << llvm::formatv("  op_desc.SetType(\"{0}\");\n", resultPatOp->getValueAsString("type"));
  
  for(auto arg : resultPattern->getArgs()) {
    StringRef argKey = dyn_cast<DefInit>(arg)->getDef()->getValueAsString("name");
    os << llvm::formatv("  op_desc.SetInput(\"{0}\", {matched.at(\"{1}\")->arg()->name});\n",
                        argKey, argKey);
  }
  
  os << "  op_desc.SetOutput(\"Out\", {matched.at(\"Out\")->arg()->name});\n";
  os << llvm::formatv("  op_desc.SetAttr(\"in_num_col_dims\", matched.at(\"{0}\")->stmt()->op_info()->GetAttr<int>(\"x_num_col_dims\"));\n",
                      patHeadOpKey);
  
  os << "  if (is_quantized_op) {\n";
  
  unsigned resOpArgIndex = 0;
  for(auto arg : firstOpDag->getArgs()) {
    auto argRec = dyn_cast<DefInit>(arg)->getDef();
    StringRef argKey = argRec->getValueAsString("name");
    os << llvm::formatv("    op_desc.SetInputScale(matched.at(\"{0}\")->arg()->name, {1}_scale_vct);\n",
                        argKey, patHeadOpArgNames[resOpArgIndex++]->getValue());
  }
  os << "  }\n";
  os << "  return op_desc;\n";
  os<< "}\n";

}

//生成源码中`InsertNewNode`这个方法。
void GraphOptPassEmitter::EmitInsertNewNodeMethod(Record* record, raw_ostream& os) {
  if (!firstOpDag) {
    PrintFatalError("firstOpDag not found.");
  }
  StringRef firstOpType = getOpFromDagInit(firstOpDag)->getValueAsString("type");
  
  DagInit* resultPat = getResultPatFromPat(record);
  Record* resultPatOp = getOpFromDagInit(resultPat);
  StringRef resultPatOpType = resultPatOp->getValueAsString("type");
  
  auto patName = record->getName().drop_back(3);
  os << "void " << patName
      << "Fuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {\n";
  
  os << "  auto op_desc = GenOpDesc(matched);\n";
  
  os << llvm::formatv("  auto {0}_op = LiteOpRegistry::Global().Create(\"{1}\");\n",
                      resultPatOpType, resultPatOpType);
  
  os << llvm::formatv("  auto {0} = matched.at(\"{1}\")->stmt()->op();\n",
                      firstOpType, firstOpType);
  
  os << llvm::formatv("  auto* scope = {0}->scope();\n", firstOpType);
  os << llvm::formatv("  auto& valid_places = {0}->valid_places();\n", firstOpType);
  os << llvm::formatv("  {0}_op->Attach(op_desc, scope);\n", resultPatOpType);
  
  os << llvm::formatv("  auto* new_op_node = graph->GraphCreateInstructNode({0}_op, valid_places);\n",
                      resultPatOpType);
  
  for(auto arg : resultPat->getArgs()) {
    StringRef argKey = dyn_cast<DefInit>(arg)->getDef()->getValueAsString("name");
    os << llvm::formatv("  IR_NODE_LINK_TO(matched.at(\"{0}\"), new_op_node);\n",
                        argKey);
  }
  
  os << "  IR_NODE_LINK_TO(new_op_node, matched.at(\"Out\"));\n";
  
  os << "}\n";
}

//生成源码中`BuildPattern`这个方法。
void GraphOptPassEmitter::EmitBuildPatternMethod(Record* record, raw_ostream& os) {
  DagInit* sourcePattern = record->getValueAsDag("sourcePattern");
  
  //将xxxPat后的Pat丢掉
  auto patName = record->getName().drop_back(3);
  os << "void " << patName << "Fuser::BuildPattern() {\n";
  
  StringRef innerOpKey = dfsPatDag(sourcePattern, os);
  os << "  auto* Out = VarNode(\"Out\");";
  os << "\n";
  
  os << llvm::formatv("  {0}_inputs >> *{1} >> *{2};",
                      innerOpKey, innerOpKey, "Out");
  os << "\n";
  
  os << "}\n";
}

void GraphOptPassEmitter::EmitFuserHeader(Record* record, raw_ostream& os) {
  
  StringRef className = "somefuser";
  
  os << R"(// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
  
  os << "\n";
  os << "#pragma once";
  os << "\n\n";
  
  os << R"(#include <memory>
#include <string>
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"
)";
  os << "\n";
  os << R"(namespace paddle {
namespace lite {
namespace mir {
namespace fusion {
)";
  
  os << "\n";
  os << "class " << className << " : public FuseBase {\n";
  os << "public:\n";
  os << "  explicit FcFuser(bool with_relu) : with_relu_(with_relu) {}\n";
  os << "  void BuildPattern() override;\n";
  os << "  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;\n";
  os << "private:\n";
  os << "  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;\n";
  os << "  bool with_relu_;\n";
  os << "};\n";
  
  os << "\n";
  
  os << R"(}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
)";

}

void GraphOptPassEmitter::run(raw_ostream &OS) {
  //emitSourceFileHeader("GraphOptPass", OS);
  auto recs = Records.getAllDerivedDefinitions("Pat");
  
  for(Record *rec : recs) {
    dbgs() << "---------------" << rec->getName() << "---------------\n";
    EmitBuildPatternMethod(rec, OS);
    EmitInsertNewNodeMethod(rec, OS);
    EmitGenOpDescMethod(rec, OS);
  }
  
}

namespace llvm {

void EmitGraphOptPass(RecordKeeper &RK, raw_ostream &OS) {
  GraphOptPassEmitter(RK).run(OS);
}

} // namespace llvm
