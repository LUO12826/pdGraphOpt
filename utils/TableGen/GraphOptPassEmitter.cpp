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
#include "llvm/Support/Format.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <string>
#include <vector>
#include "PdLiteGraphOpt/RecordConverter.h"

#define DEBUG_TYPE "graph-opt-pass-emitter"

using namespace llvm;

namespace {

//对于每个pass生成4部分，内容分别对应原来的.h file and .cpp file for pass class and fuser class.
class GraphOptPassEmitter {
  
private:
  RecordKeeper &Records;

  std::map<std::string, unsigned> opKeyAllocationRecord;

  std::map<std::string, unsigned> varKeyAllocationRecord;

  std::map<PdGraphOpt::TDOperator*, std::string> op2key;

  std::string srcPatOutputKey{"Out"};

  std::string registerOp(PdGraphOpt::TDOperator* op) {
    if(op2key.count(op)) {
      llvm::PrintFatalError("Do not register a op twice.");
    }
    else {
      std::string key = getNewOpKey(op->getType());
      op2key.insert(std::make_pair(op, key));
      return key;
    }
  }

  std::string getNewOpKey(std::string opType) {
    if (opKeyAllocationRecord.count(opType)) {
      unsigned count = opKeyAllocationRecord.at(opType) + 1;
      opKeyAllocationRecord.at(opType)++;
      return opType + "_" + std::to_string(count);
    }
    else {
      opKeyAllocationRecord.insert(std::make_pair(opType, 0));
      return opType + "_" + std::to_string(0);
    }
  }

  void EmitFuserHeader(Record*, raw_ostream&);
  
  //void EmitFuserImpl(Record*, raw_ostream&);

  void EmitBuildPatternMethod(PdGraphOpt::TDPattern &pat, raw_ostream&);

  void EmitInsertNewNodeMethod(PdGraphOpt::TDPattern &pat, raw_ostream& os);

  void EmitGenOpDescMethod(PdGraphOpt::TDPattern &pat, raw_ostream& os);
  
  //void EmitPassHeader(Record*, raw_ostream&);
  
  //void EmitPassImpl(Record*, raw_ostream&);

  std::string dfsPatDag(PdGraphOpt::TDPatternOpNode *dag, raw_ostream& os);
  
  ///Pattern（有向图）的首个节点，融合时会作为替换点位
  DagInit* firstOpDag{nullptr};

  PdGraphOpt::TDPatternOpNode *leadingOp{nullptr};

public:
  GraphOptPassEmitter(RecordKeeper &RK) : Records(RK) {}

  void run(raw_ostream &OS);
  
};

} // anonymous namespace

//辅助函数
namespace {

//从一个Pat中获取resultPattern的便捷方法
//DagInit* getResultPatFromPat(Record* record) {
//  auto resultPatternsArr = record->getValueAsListInit("resultPatterns")->getValues();
//  if (resultPatternsArr.size() < 1) {
//    PrintFatalError("Pat has no result pattern.");
//  }
//  return dyn_cast<DagInit>(resultPatternsArr[0]);
//}

//从DagInit中获取dag的operator的便捷方法
//Record* getOpFromDagInit(DagInit* dag) {
//  auto opInit = dyn_cast<DefInit>(dag->getOperator());
//  if(opInit) {
//    return opInit->getDef();
//  } else {
//    PrintFatalError("No op or op is not a record.");
//  }
//}

} // anonymous namespace


//遍历sourcePattern，生成定义VarNode、OpNode及其定义它们拓扑关系的代码
std::string GraphOptPassEmitter::dfsPatDag(PdGraphOpt::TDPatternOpNode *dag, raw_ostream& os) {

  auto op = dag->getOp();
  std::string opType = op->getType();
  std::string opKey = registerOp(op.get());
  auto &args = dag->getArguments();
  auto &argNames = dag->getArgNames();

  //生成该dag的op对应的OpNode
  os << "  //Start op " << opKey << "\n";
  os << llvm::formatv("  auto* {0} = OpNode(\"{1}\", \"{2}\");",
                      opKey, opKey, opType);
  os << "\n";
  os << "  " << opKey << "->AsIntermediate();";
  os << "\n\n";


  //检查这个dag里面是否还嵌套着另一个dag
  bool patWithoutDag = true;
  for(auto &arg : args) {
    if (arg->getNodeType() == PdGraphOpt::TDPatternNode::NodeType::Op) {
      patWithoutDag = false;
      break;
    }
  }

  //这个OpNode的input
  std::vector<std::string> inputs;

  //如果这个dag没有嵌套的dag，生成它的VarNode时会生成一个assert_is_op_input调用。
  if (patWithoutDag) {
    this->leadingOp = dag;
    auto opArgNames = op->getArgNames();

    unsigned currentArg = 0;
    for(auto &argName : argNames) {
      inputs.push_back(argName);

      os << llvm::formatv("  auto* {0} = VarNode(\"{1}\")->assert_is_op_input(\"{2}\", \"{3}\");",
                          argName, argName, opType, opArgNames[currentArg++]);
      os << "\n";
    }
  }
  //如果这个dag有嵌套的dag，生成它的VarNode时会生成一个assert_is_persistable_var调用。
  //目前这样的处理仅仅是根据paddle中FcFuser得出的。
  else {

    for(unsigned i = 0, c = args.size(); i < c; i++) {
      //如果这个dag的参数是DefInit，也就是说它是个Var，所以生成一个VarNode
      if (args[i]->getNodeType() == PdGraphOpt::TDPatternNode::NodeType::Var) {
        auto argPtr = args[i].get();
        auto varArgPtr = static_cast<PdGraphOpt::TDPatternVarNode*>(argPtr);
        if (varArgPtr->getVar()->getType() == "tensor") {
          std::string varName = argNames[i];
          inputs.push_back(varName);
          os << llvm::formatv("  auto* {0} = VarNode(\"{1}\")->assert_is_persistable_var();",
                              varName, varName);
          os << "\n";
        }
      }
      //如果这个dag的参数是DagInit，也就是说它是个Dag，所以递归调用本方法。
      else if(args[i]->getNodeType() == PdGraphOpt::TDPatternNode::NodeType::Op) {
        auto argPtr = args[i].get();
        auto opArgPtr = static_cast<PdGraphOpt::TDPatternOpNode*>(argPtr);

        std::string innerOpKey = dfsPatDag(opArgPtr, os);
        std::string innerOpOutKey = innerOpKey + "_out";

        inputs.push_back(innerOpOutKey);

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
  std::string opInputSet = opKey + "_inputs";
  os << "  std::vector<PMNode*> " << opInputSet << " {";
  bool first = true;
  for(std::string &input : inputs) {
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
void GraphOptPassEmitter::EmitGenOpDescMethod(PdGraphOpt::TDPattern &pat, raw_ostream& os) {
  if (!leadingOp) {
    PrintFatalError("leadingOp not found.");
  }

  auto patLeadingOp = leadingOp->getOp();

  os << "cpp::OpDesc " << pat.getNameWithoutPat() << "Fuser::GenOpDesc(const key2nodes_t& matched) {\n";

  std::string patHeadOpKey = op2key.at(patLeadingOp.get());
  os << llvm::formatv("  auto op_desc = *matched.at(\"{0}\")->stmt()->op_info();\n", patHeadOpKey);

  auto patHeadOpArgNames = patLeadingOp->getArgNames();
  for(auto &arg : patHeadOpArgNames) {
    os << llvm::formatv("  auto input_{0}_name = op_desc.Input(\"{1}\").front();\n", arg, arg);
    os << llvm::formatv("  std::vector<float> {0}_scale_vct;\n", arg);
  }

  if (patHeadOpArgNames.size() == 1) {
    os << llvm::formatv("  bool is_quantized_op = op_desc.HasInputScale(input_{0}_name);\n",
                        patHeadOpArgNames.front());
  }
  else {
    for(size_t i = 0, argCount = patHeadOpArgNames.size(); i < argCount - 1; i++) {
      std::string argName = patHeadOpArgNames[i];
      if (i == 0) {
        os << llvm::formatv("  bool is_quantized_op = op_desc.HasInputScale(input_{0}_name) &&\n", argName);
      }
      else {
        os << llvm::formatv("                         op_desc.HasInputScale(input_{0}_name) &&\n", argName);
      }
    }
    os << llvm::formatv("                         op_desc.HasInputScale(input_{0}_name);\n", patHeadOpArgNames.back());
  }

  os << "  if (is_quantized_op) {\n";
  for(auto &argName : patHeadOpArgNames) {
    os << llvm::formatv("    {0}_scale_vct = op_desc.GetInputScale(input_{1}_name);\n", argName, argName);
  }
  os << "  }\n";

  os << "  op_desc.mutable_inputs()->clear();\n";
  os << "  op_desc.mutable_outputs()->clear();\n";

  auto targetPattern = pat.getTargetPatternRoot();
  auto targetPatOp = targetPattern->getOp();

  os << llvm::formatv("  op_desc.SetType(\"{0}\");\n", targetPatOp->getType());

  for(auto &argName : targetPattern->getArgNames()) {
    os << llvm::formatv("  op_desc.SetInput(\"{0}\", {matched.at(\"{1}\")->arg()->name});\n",
                        argName, argName);
  }

  os << llvm::formatv(
      "  op_desc.SetOutput(\"{0}\", {matched.at(\"{1}\")->arg()->name});\n",
      targetPatOp->getResNames()[0], srcPatOutputKey);
  os << llvm::formatv("  op_desc.SetAttr(\"in_num_col_dims\", matched.at(\"{0}\")->stmt()->op_info()->GetAttr<int>(\"x_num_col_dims\"));\n",
                      patHeadOpKey);

  os << "  if (is_quantized_op) {\n";

  unsigned resOpArgIndex = 0;
  for(auto &argName : leadingOp->getArgNames()) {
    os << llvm::formatv("    op_desc.SetInputScale(matched.at(\"{0}\")->arg()->name, {1}_scale_vct);\n",
                        argName, patHeadOpArgNames[resOpArgIndex++]);
  }
  os << "  }\n";
  os << "  return op_desc;\n";
  os<< "}\n";

}

//生成源码中`InsertNewNode`这个方法。
void GraphOptPassEmitter::EmitInsertNewNodeMethod(PdGraphOpt::TDPattern &pat, raw_ostream& os) {
  if (!leadingOp) {
    PrintFatalError("leadingOp not found.");
  }
  std::string leadingOpType = leadingOp->getOp()->getType();
  std::string leadingOpKey = op2key.at(leadingOp->getOp().get());
  
  auto targetPat = pat.getTargetPatternRoot();
  auto targetPatOp = targetPat->getOp();
  std::string targetPatOpType = targetPatOp->getType();
  std::string targetPatOpKey = registerOp(targetPatOp.get());

  os << "void " << pat.getNameWithoutPat()
      << "Fuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {\n";
  
  os << "  auto op_desc = GenOpDesc(matched);\n";
  
  os << llvm::formatv("  auto {0}_op = LiteOpRegistry::Global().Create(\"{1}\");\n",
                      targetPatOpKey, targetPatOpType);
  
  os << llvm::formatv("  auto {0} = matched.at(\"{1}\")->stmt()->op();\n",
                      leadingOpKey, leadingOpKey);
  
  os << llvm::formatv("  auto* scope = {0}->scope();\n", leadingOpKey);
  os << llvm::formatv("  auto& valid_places = {0}->valid_places();\n",
                      leadingOpKey);
  os << llvm::formatv("  {0}_op->Attach(op_desc, scope);\n", targetPatOpKey);
  
  os << llvm::formatv("  auto* new_op_node = graph->GraphCreateInstructNode({0}_op, valid_places);\n",
                      targetPatOpKey);
  
  for(auto &argName : targetPatOp->getArgNames()) {
    os << llvm::formatv("  IR_NODE_LINK_TO(matched.at(\"{0}\"), new_op_node);\n",
                        argName);
  }

  os << llvm::formatv(
      "  IR_NODE_LINK_TO(new_op_node, matched.at(\"{0}\"));\n", srcPatOutputKey);
  os << "}\n";
}

//生成源码中`BuildPattern`这个方法。
void GraphOptPassEmitter::EmitBuildPatternMethod(PdGraphOpt::TDPattern &pat, raw_ostream& os) {

  auto *srcPatRoot = pat.getSourcePatternRoot();

  os << "void " << pat.getNameWithoutPat() << "Fuser::BuildPattern() {\n";

  std::string innerOpKey = dfsPatDag(srcPatRoot, os);
  this->srcPatOutputKey = srcPatRoot->getOp()->getResNames()[0];
  os << llvm::formatv("  auto* {0} = VarNode(\"{1}\");\n",
                      srcPatOutputKey, srcPatOutputKey);

  os << llvm::formatv("  {0}_inputs >> *{1} >> *{2};\n",
                      innerOpKey, innerOpKey, srcPatOutputKey);

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

  PdGraphOpt::RecordConverter converter(Records);
  std::vector<PdGraphOpt::TDPattern> patterns = converter.getPatterns();

  for(PdGraphOpt::TDPattern &pat : patterns) {
    dbgs() << "---------------" << pat.getName() << "---------------\n";
    EmitBuildPatternMethod(pat, OS);
    EmitGenOpDescMethod(pat, OS);
    EmitInsertNewNodeMethod(pat, OS);
  }
  
}

namespace llvm {

void EmitGraphOptPass(RecordKeeper &RK, raw_ostream &OS) {
  GraphOptPassEmitter(RK).run(OS);
}

} // namespace llvm
