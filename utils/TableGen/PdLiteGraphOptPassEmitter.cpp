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

#include "PdLiteGraphOpt/RecordConverter.h"
#include "PdLiteGraphOpt/CodeFragment.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <string>
#include <vector>

#define DEBUG_TYPE "pd-lite-graph-opt-pass-emitter"

using namespace llvm;
using namespace PdGraphOpt;

namespace PdGraphOpt {
enum ComparisonOperator {
  GreaterThan = 0,
  Equals = 1,
  LessThan = 2
};

struct ArgumentKeySourceInfo {
  std::string key;
  std::string opKey;

  long index;
  std::string slotName;

  TDOpArgument::ArgumentType argType;
  std::string dataType;
};

}

namespace PdGraphOpt {

// 生成fusion pass
class PdLiteGraphOptPassEmitter {

private:
  RecordKeeper &Records;

  //是否输出为单文件
  bool singleFileMode{true};

  //把op按类型计数。用于在给op分配key时生成唯一字符串。
  std::map<std::string, unsigned> opKeyAllocationRecord;

  // op到它被分配的key间的映射。
  std::map<PdGraphOpt::TDPatternOpNode *, std::string> op2key;

  // op的key到op间的映射。
  std::map<std::string, PdGraphOpt::TDPatternOpNode *> key2op;

  //用户给op设定的key（如果有的话）到op被分配的key间的映射。
  std::map<std::string, std::string> opDesignatedKey2opKey;

  // argument的key到其所属的op的映射。
  std::map<std::string, std::string> argKeyToSrcPatOpKey;

  std::vector<std::string> srcPatOutputNames{"Out"};

  /// Pattern（有向图）的首个节点，融合时会作为替换点位
  PdGraphOpt::TDPatternOpNode *leadingOpNode{nullptr};

  /// 生成了一个pass之后，重置状态
  void resetState() {
    leadingOpNode = nullptr;
    srcPatOutputNames = {"Out"};
    argKeyToSrcPatOpKey.clear();
    opDesignatedKey2opKey.clear();
    opKeyAllocationRecord.clear();
    key2op.clear();
    op2key.clear();
  }

  //生成对一个算子输入参数的维度进行检查的代码
  void EmitInputDimChecker(std::vector<std::string> inputNames,
                           std::vector<int> dims,
                           std::vector<ComparisonOperator> compOp,
                           raw_ostream &os
                           ) {
    os << "[](const Node* node) -> bool {\n";
    os << "  auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();\n";
    for (auto &name : inputNames) {
      os << llvm::formatv("  auto input_{0}_name = op_desc.Input(\"{1}\").front();\n",
                          name, name);
    }
    os << "  auto* scope = const_cast<Node*>(node)->AsStmt().op()->scope();\n";
    for (auto &name : inputNames) {
      os << llvm::formatv("  size_t {0}_rank = scope->FindVar(input_{1}_name)->Get<lite::Tensor>().dims().size();\n",
                          name, name);
    }

    os << "  bool res = true;\n";

    bool assertAllEqual = true;
    for(int dim : dims) {
      if (dim != 0) assertAllEqual = false;
    }
    static std::string compOps[] = {">", "==", "<"};
    if (assertAllEqual) {
      for (unsigned i = 1; i < inputNames.size(); i++) {
        os << llvm::formatv("  res &= {0}_rank == {1}_rank;\n",
                            inputNames[i - 1],
                            inputNames[i]);
      }
    }
    else {
      for (unsigned i = 0; i < inputNames.size(); i++) {
        os << llvm::formatv("  res &= {0}_rank {1} {2};\n",
                            inputNames[i],
                            compOps[compOp[i]],
                            dims[i]);
      }
    }

    os << "  return res;\n";
    os << "}\n";
  }

  void EmitSingleFileHeader(raw_ostream &os);

  void EmitFuserHeader(PdGraphOpt::TDPattern &pat, raw_ostream &os);

  void EmitFuserImpl(PdGraphOpt::TDPattern &pat, raw_ostream &os);

  void EmitBuildPatternMethod(PdGraphOpt::TDPattern &pat, raw_ostream &os);

  void EmitBuildNewGraphMethod(PdGraphOpt::TDPattern &pat, raw_ostream &os);

  void EmitPassHeader(PdGraphOpt::TDPattern &pat, raw_ostream &os);

  void EmitPassImpl(PdGraphOpt::TDPattern &pat, raw_ostream &os);

  void EmitRegisterPass(PdGraphOpt::TDPattern &pat, raw_ostream &os);

  std::string dfsSrcPatDag(PdGraphOpt::TDPatternOpNode *dag, raw_ostream &os,
                        PdGraphOpt::TDPattern &pat);

  std::string dfsResPatDagV2(PdGraphOpt::TDPatternOpNode *dag, bool isRoot,
                           raw_ostream &os, PdGraphOpt::TDPattern &pat);

  void handleTargetPatTopological(PdGraphOpt::TDPattern &pat,
                                  raw_ostream &os);

  std::pair<std::string, std::string>
    handleDirectComputeV2(PdGraphOpt::TDPatternOpNode *dag, TDPattern &pat, raw_ostream& os);

  std::string registerOp(PdGraphOpt::TDPatternOpNode *opNode);

  bool getKeySourceInfoByKey(std::string key,ArgumentKeySourceInfo &info);

  std::string getNewOpKey(std::string opType);

public:
  PdLiteGraphOptPassEmitter(RecordKeeper &RK) : Records(RK) {}

  void run(raw_ostream &OS);
};

/**
 * 将一个std::vector<std::string>转为字符串列表字面量
 */
std::string genListString(std::vector<std::string> &list, std::string separator,
                          std::string head, std::string tail) {
  std::stringstream ss;
  ss << head << " ";
  bool first = true;
  for (std::string &item : list) {
    if (first) {
      ss << item;
      first = false;
    } else
      ss << separator << " " << item;
  }
  ss << " " << tail;
  return ss.str();
}

/**
 * 将驼峰命名法转为蛇形命名法
 */
std::string convertCamelToSnake(std::string camel) {
  if (camel.empty()) return "";
  std::stringstream ss;

  ss << (char)tolower(camel[0]);
  for (unsigned i = 1; i < camel.size(); i++) {
    char c = camel[i];
    if (isupper(c)) {
      ss << "_" << (char)tolower(c);
    }
    else {
      ss << c;
    }
  }
  return ss.str();
}

} // PdGraphOpt namespace

std::string
PdLiteGraphOptPassEmitter::registerOp(PdGraphOpt::TDPatternOpNode *opNode) {
  if (op2key.count(opNode)) {
    llvm::PrintFatalError("Do not register a op twice.");
  } else {
    std::string key = getNewOpKey(opNode->getOp()->getType());
    op2key.insert(std::make_pair(opNode, key));
    key2op.insert(std::make_pair(key, opNode));
    if (opNode->getOp()->getKey() != "") {
      opDesignatedKey2opKey.insert(
          std::make_pair(opNode->getOp()->getKey(), key));
    }
    return key;
  }
}

std::string PdLiteGraphOptPassEmitter::getNewOpKey(std::string opType) {
  if (opKeyAllocationRecord.count(opType)) {
    opKeyAllocationRecord[opType]++;
    return opType + "_" + std::to_string(opKeyAllocationRecord[opType]);
  } else {
    opKeyAllocationRecord[opType] = 0;
    return opType + "_" + std::to_string(0);
  }
}

void genAttrAssertCode(const AttrToAssert& attr, std::string opKey, raw_ostream &os) {
  if (attr.useCustomAssert) {
    os << llvm::formatv(
        "  {0}->assert_op_attr_satisfied<{1}>(\"{2}\", {3});\n", opKey,
        attr.dataType, attr.attrName, attr.customAssert);
  }
  //对于浮点数，考虑误差
  else if (attr.dataType == "float" || attr.dataType == "double") {
    os << llvm::formatv(
        "  {0}->assert_op_attr_satisfied<{1}>(\"{2}\", "
        "[]({3} attr) { return (std::fabs(attr - {4}) < 1e-5); });\n",
        opKey, attr.dataType, attr.attrName, attr.dataType, attr.value);
  }
  else {
    os << llvm::formatv(
        "  {0}->assert_op_attr<{1}>(\"{2}\", {3});\n", opKey, attr.dataType,
        attr.attrName,
        attr.dataType != "string" ? attr.value : "\"" + attr.value + "\"");
  }
}

/// 生成源码中的`BuildPattern`方法。这个方法是生成代码时最先被调用的。
void PdLiteGraphOptPassEmitter::EmitBuildPatternMethod(
    PdGraphOpt::TDPattern &pat, raw_ostream &os) {
  auto *srcPatRoot = pat.getSourcePatternRoot();

  os << "void " << pat.getNameWithoutPat() << "Fuser::BuildPattern() {\n";

  std::string innerOpKey = dfsSrcPatDag(srcPatRoot, os, pat);

  std::vector<std::string> srcPatOutputNames;
  for (auto &name : srcPatRoot->getOp()->getResNames()) {
    srcPatOutputNames.push_back(innerOpKey + name);
  }
  this->srcPatOutputNames = srcPatOutputNames;
  std::string innerOpType = srcPatRoot->getOp()->getTypeAuto();

  bool first = true;
  for (unsigned i = 0; i < srcPatOutputNames.size(); i++) {
    os << llvm::formatv(
        "  auto* {0} = VarNode(\"{1}\")->assert_is_op_output({2}, \"{3}\");\n",
        srcPatOutputNames[i], srcPatOutputNames[i], innerOpType,
        srcPatRoot->getOp()->getResNames()[i]);
    if (!first) {
      os << "->AsIntermediate()";
    } else {
      first = false;
    }
    os << ";\n";
  }
  os << llvm::formatv("  std::vector<PMNode*> {0}_outputs{1};\n",
                      innerOpKey,
                      genListString(this->srcPatOutputNames,",", "{", "}"));
  os << llvm::formatv("  {0}_inputs >> *{1} >> {0}_outputs;\n", innerOpKey,
                      innerOpKey, innerOpKey);

  os << "}\n";
}

//遍历sourcePattern，生成定义VarNode、OpNode及其定义它们拓扑关系的代码
//在EmitBuildPatternMethod中被调用
std::string
PdLiteGraphOptPassEmitter::dfsSrcPatDag(TDPatternOpNode *dag, raw_ostream &os,
                                     TDPattern &pat) {

  auto op = dag->getOp();
  std::string opType = op->getTypeAuto();

  std::string opKey = registerOp(dag);
  // TODO: 处理op的输出被绑定了key以及op多输出的情况

  //生成该dag的op对应的OpNode
  os << "  //Start op " << opKey << "\n";
  os << llvm::formatv("  auto* {0} = OpNode(\"{1}\", {2});", opKey, opKey,
                      opType);
  os << "\n";
  os << "  " << opKey << "->AsIntermediate();\n";

  //为这个OpNode添加断言
  if (pat.getAttrsToAssert().count(op->getKey())) {
    auto &attrs = pat.getAttrsToAssert().at(op->getKey());
    for (auto &&attr : attrs) {
      genAttrAssertCode(attr, opKey, os);
    }
  }
  if (pat.getCustomTellers().count(op->getKey())) {
    auto &tellers = pat.getCustomTellers().at(op->getKey());
    for (auto &&tel : tellers) {
      os << llvm::formatv("  {0}->assert_node_satisfied({1});\n", opKey,
                          tel.teller);
    }
  }

  os << "\n";

  //检查这个dag里面是否还嵌套着另一个dag
  //检查这个dag里面是否有attribute，有的话当做断言处理
  auto &args = dag->getArguments();
  bool patWithoutDag = true;
  for (unsigned i = 0; i < args.size(); i++) {
    auto arg = args[i].get();
    auto nodeType = arg->getNodeType();
    if (nodeType == TDPatternNode::Op) {
      patWithoutDag = false;
    }
    if (nodeType == TDPatternNode::Attr) {
      TDPatternAttrNode* node = (TDPatternAttrNode*)arg;
      if (node->getAttr() != nullptr) {
        AttrToAssert newAssert;
        newAssert.target = op->getKey();
        newAssert.value = node->getAttr()->getValue();
        newAssert.dataType = node->getAttr()->getDataType();
        newAssert.attrName = op->getArgumentAsAttrAtIndex(i)->getName();
        genAttrAssertCode(newAssert, opKey, os);
      }

    }
  }
  if (patWithoutDag) {
    this->leadingOpNode = dag;
  }

  //这个op的参数名称列表，参数名用来作为VarNode的变量名。
  std::vector<std::string> inputs;
  //这个op固有的参数名称列表，即这个op在td中定义时书写的参数名称列表
  auto opArgNames = op->getArgNames();
  //这个op在pattern中的参数名称列表
  auto &argNames = dag->getArgNames();

  //处理一个OpNode的参数，生成对应的VarNode
  for (unsigned i = 0, c = args.size(); i < c; i++) {
    //如果这个dag的参数是个Var，生成一个VarNode
    if (args[i]->getNodeType() == TDPatternNode::Var) {
      //这个Var的名称（或者说是key，dag中指定的实际参数名称），作为生成VarNode时的变量名
      std::string argKey = argNames[i];
      inputs.push_back(argKey);
      argKeyToSrcPatOpKey[argKey] = opKey;
      os << llvm::formatv(
          "  auto* {0} = VarNode(\"{1}\")->assert_is_op_input({2}, "
          "\"{3}\")",
          argKey, argKey, opType, opArgNames[i]);

      auto varArgPtr =
          static_cast<TDPatternVarNode *>(args[i].get());
      if (varArgPtr->getVar() && varArgPtr->getVar()->getIsWeight()) {
        os << "->assert_is_persistable_var()";
      }
      if (!pat.isVarDirectlyUsedByTargetPattern(argKey)) {
        os << "->AsIntermediate()";
      }
      os << ";\n";
    }

    else if (args[i]->getNodeType() == TDPatternNode::Attr) {
      std::string argKey = argNames[i];
      //inputs.push_back(argKey);
      argKeyToSrcPatOpKey[argKey] = opKey;
    }

    //如果有嵌套的op（dag），递归处理
    else if (args[i]->getNodeType() == TDPatternNode::Op) {

      auto opArgPtr = static_cast<TDPatternOpNode *>(args[i].get());
      std::string innerOpKey = dfsSrcPatDag(opArgPtr, os, pat);
      std::string innerOpType = opArgPtr->getOp()->getTypeAuto();

      std::vector<std::string> outputs;
      auto &innerOpResNames = opArgPtr->getOp()->getResNames();
      for (unsigned j = 0, cnt = innerOpResNames.size(); j < cnt; j++) {
        std::string innerOpOutKey = innerOpKey + "_" + innerOpResNames[j];
        outputs.push_back(innerOpOutKey);
        os << llvm::formatv(
            "  auto* {0} = "
            "VarNode(\"{1}\")->assert_is_op_output({2}, \"{3}\");\n",
            innerOpOutKey, innerOpOutKey, innerOpType, innerOpResNames[j]);
        os << "  " << innerOpOutKey << "->AsIntermediate();\n";

        if (j == 0)
          inputs.push_back(innerOpOutKey);
      }

      std::string opOutputSet = innerOpKey + "_outputs";
      os << "  std::vector<PMNode*> " << opOutputSet;
      os << genListString(outputs, ",", "{", "}");
      os << ";\n";

      //连接op的输入和输出。
      os << llvm::formatv("  {0}_inputs >> *{1} >> {2};", innerOpKey,
                          innerOpKey, opOutputSet);
      os << "\n";
    }
  }

  //生成该op的input集合
  std::string opInputSet = opKey + "_inputs";
  os << "  std::vector<PMNode*> " << opInputSet;
  os << genListString(inputs, ",", "{", "}");
  os << ";\n";
  os << "  //End op " << opKey << "\n";
  return opKey;
}

void genSetAttrCode(std::string opKey, const AttrToSet &attr, raw_ostream &os) {
  if (attr.dataType == "string") {
    os << llvm::formatv(
        "  op_desc.SetAttr(\"{0}\", std::string(\"{1}\"));\n",
        attr.attrName, attr.value);
  } else {
    os << llvm::formatv("  op_desc.SetAttr(\"{0}\", {1});\n", attr.attrName,
                        attr.value);
  }
}

/// 生成源码中的`InsertNewNode`方法
void PdLiteGraphOptPassEmitter::EmitBuildNewGraphMethod(TDPattern &pat,
                                                        raw_ostream &os) {
  assert(leadingOpNode != nullptr && "leadingOpNode not found.");

  std::string leadingOpKey = op2key.at(leadingOpNode);

  auto targetPat = pat.getTargetPatternRoot();

  os << "void " << pat.getNameWithoutPat()
     << "Fuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {\n";

  os << llvm::formatv("  auto {0} = matched.at(\"{1}\")->stmt()->op();\n",
                      leadingOpKey, leadingOpKey);

  os << llvm::formatv("  auto* scope = {0}->scope();\n", leadingOpKey);
  os << llvm::formatv("  auto& valid_places = {0}->valid_places();\n",
                      leadingOpKey);

  handleTargetPatTopological(pat, os);

  os << "}\n";
}

std::string PdLiteGraphOptPassEmitter::dfsResPatDagV2(
    PdGraphOpt::TDPatternOpNode *dag,
    bool isRoot,
    raw_ostream &os,
    PdGraphOpt::TDPattern &pat) {

  std::string targetPatOpType = dag->getOp()->getType();
  std::string targetPatOpKey = registerOp(dag);

  os << llvm::formatv("  cpp::OpDesc {0}_desc;\n", targetPatOpKey);

  //开始处理SetInputScale的部分。
  //由于SetInputScale的代码有分开的两部分，这里先把后面那部分一并生成。
  std::stringstream SetInputScaleStr;
  SetInputScaleStr << "  if (is_quantized_op) {\n";
  std::stringstream GetInputScaleStr;
  GetInputScaleStr << "  if (is_quantized_op) {\n";

  // 判断是否需要拷贝InputScale
  bool needCopyInputScale = false;
  // TODO: 完善inputScale来自多个op的情况
  std::vector<AttrToCopy> inputScaleCopies;

  if (pat.getNeedCopyInputScale()) {
    for (auto &&attr : pat.getAttrsToCopy()) {
      if (attr.attrName == "#INPUT_SCALE"
          && attr.to == dag->getOp()->getKey()) {
        needCopyInputScale = true;
        inputScaleCopies.push_back(attr);
      }
    }
  }

  if (needCopyInputScale) {

    for (auto &&theAttr : inputScaleCopies) {
      std::string thatOpKey = opDesignatedKey2opKey[theAttr.from];
      auto thatDag = key2op[thatOpKey];
      auto thatOp = key2op[thatOpKey]->getOp();

      std::vector<std::string> inputSlotNames;
      std::vector<unsigned > inputSlotIndices;

      for (unsigned i = 0; i < thatOp->getArgNames().size(); i++) {
        if (thatOp->getArgumentTypeAtIndex(i) == TDOpArgument::variable) {
          inputSlotNames.push_back(thatOp->getArgNames()[i]);
          inputSlotIndices.push_back(i);
        }
      }

      os << llvm::formatv(
          "  auto* {0}_desc = matched.at(\"{1}\")->stmt()->op_info();\n",
          thatOpKey, thatOpKey);
      os << llvm::formatv(
          "  bool {0}_is_quantized_op = true;\n", thatOpKey);

      for (unsigned i = 0; i < inputSlotNames.size(); i++) {
        std::string &inputSlot = inputSlotNames[i];
        std::string formArgName = inputSlot;
        os << llvm::formatv(
            "  auto {0}_input_{1}_name = {2}_desc.Input(\"{3}\").front();\n",
            thatOpKey, inputSlot, thatOpKey, inputSlot);
        os << llvm::formatv("  std::vector<float> {0}_{1}_scale_vct;\n",
                            thatOpKey, inputSlot);

        os << llvm::formatv(
            "  {0}_is_quantized_op &= {1}_desc.HasInputScale({2}_input_{3}_name);\n",
            thatOpKey, thatOpKey, thatOpKey, inputSlot);

        GetInputScaleStr << llvm::formatv(
                                "    {0}_{1}_scale_vct = op_desc.GetInputScale({2}_input_{3}_name);\n",
                                thatOpKey, inputSlot, thatOpKey, inputSlot)
                                .str();

        std::string actualName;
        if (inputSlotIndices[i] < thatDag->getArgNames().size()) {
          actualName = thatDag->getArgNames()[inputSlotIndices[i]];
        }
        else {
          PrintFatalError("");
        }

        SetInputScaleStr
            << llvm::formatv(
                   "    {0}_desc.SetInputScale(matched.at(\"{1}\")->arg()->name, "
                   "{2}_{3}_scale_vct);\n",
                   targetPatOpKey, actualName, thatOpKey, inputSlot)
                   .str();
      }
      SetInputScaleStr << "  }\n";
      GetInputScaleStr << "  }\n";

      os << GetInputScaleStr.str();
    }

  } //结束处理SetInputScale的部分。

  //  os << "  op_desc.mutable_inputs()->clear();\n";
  //  os << "  op_desc.mutable_outputs()->clear();\n";

  auto targetPattern = dag;
  auto targetPatOp = targetPattern->getOp();
  auto &targetPatArgSlotNames = targetPatOp->getArgNames();
  auto &targetPatArgNames = targetPattern->getArgNames();
  auto &targetPatArgs = targetPattern->getArguments();

  os << llvm::formatv("  {0}_desc.SetType(\"{1}\");\n",
                      targetPatOpKey, targetPatOp->getType());

  unsigned actualArgSize = targetPatArgNames.size();
  std::stringstream IR_NODE_LINK_TO_code;

  for (unsigned i = 0; i < targetPatArgs.size(); i++) {
    //检查形参和实参类型是否匹配
    if (!targetPattern->isArgTypeCorrect(i)) {
      PrintFatalError("");
    }

    std::string argName = targetPatArgNames[i];
    bool isRefComputeResult =
        pat.getTargetOpByDesignatedKey(argName) != nullptr;
    if (targetPatArgs[i]->getNodeType() == TDPatternNode::Var && !isRefComputeResult) {
      os << llvm::formatv(
          "  {0}_desc.SetInput(\"{1}\", {matched.at(\"{2}\")->arg()->name});\n",
          targetPatOpKey, targetPatArgSlotNames[i], argName);

      IR_NODE_LINK_TO_code << llvm::formatv(
                                  "  IR_NODE_LINK_TO(matched.at(\"{0}\"), {1}_op_node);\n",
                                  argName, targetPatOpKey).str();
    }
    else if (targetPatArgs[i]->getNodeType() == TDPatternNode::Var && isRefComputeResult) {
      std::string newNodeName = targetPatOpKey + "_" + targetPatArgSlotNames[i];
      os << llvm::formatv("  auto* {0}_node = graph->NewArgumentNode(\"{1}\");\n",
                          newNodeName, newNodeName);
      if (targetPatOp->getArgumentTypeAtIndex(i) == TDOpArgument::variable
          && targetPatOp->getArgumentAsVarAtIndex(i)->getIsWeight()) {
        os << llvm::formatv("  {0}_node->arg()->is_weight = true;\n",
                            newNodeName);
      }
      os << llvm::formatv("  {0}_node->arg()->type = LiteType::GetTensorTy(TARGET(kUnk), PRECISION(kFloat), DATALAYOUT(kUnk));\n",
                          newNodeName);
      os << llvm::formatv("  auto* {0}_t = scope->NewTensor(\"{1}\");\n",
                          newNodeName, newNodeName);
      os << llvm::formatv("  {0}_t->set_precision(paddle::lite_api::PrecisionType::kFloat);\n",
                          newNodeName);

      os << llvm::formatv("  {0}_t->CopyDataFrom({1});\n",
                          newNodeName, argName + "_val");
      os << llvm::formatv(
          "  {0}_desc.SetInput(\"{1}\", {{\"{2}\"});\n",
          targetPatOpKey, targetPatArgSlotNames[i], newNodeName);

      IR_NODE_LINK_TO_code << llvm::formatv(
                                  "  IR_NODE_LINK_TO({0}_node, {1}_op_node);\n",
                                  newNodeName, targetPatOpKey).str();
    }
    else if (targetPatArgs[i]->getNodeType() == TDPatternNode::Attr) {
      continue;
    }
    // 还未实现targetPattern里有多个算子的情况
    // 这种情况不仅需要创建新的op node，还需要创建新的var node。
    // 但如何创建var node尚存疑问。

    // 目前只处理directCompute
    else if (targetPatArgs[i]->getNodeType() == TDPatternNode::Op) {

      auto innerOpNode = static_cast<TDPatternOpNode*>(targetPatArgs[i].get());

      std::string innerOpKey;
      if (innerOpNode->getDesignatedOutputKey() != "") {
        innerOpKey = innerOpNode->getDesignatedOutputKey();
      }
      else {
        innerOpKey = op2key[innerOpNode];
      }

      if (innerOpNode->getOp()->getType() == "DirectCompute") {
        //现在已经完成了参数计算，得到了一个pair
        //第一个值是计算结果的变量名（还需要在后面加上_val）
        //第二个值是计算结果的类型

        //创建var node和新Tensor实例代码
        //        std::string fusion_bias_name = filter_name + "_conv_fusion_bias";
        //        auto* fusion_bias_node = graph->NewArgumentNode(fusion_bias_name);
        //        fusion_bias_node->arg()->is_weight = true;
        //        fusion_bias_node->arg()->type = LiteType::GetTensorTy(
        //            TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
        //
        //        auto* fusion_bias_t = scope->NewTensor(fusion_bias_name);
        //        fusion_bias_t->set_precision(paddle::lite_api::PrecisionType::kFloat);

        std::string newNodeName = targetPatOpKey + "_" + targetPatArgSlotNames[i];
        os << llvm::formatv("  auto* {0}_node = graph->NewArgumentNode(\"{1}\");\n",
                            newNodeName, newNodeName);
        if (targetPatOp->getArgumentTypeAtIndex(i) == TDOpArgument::variable
            && targetPatOp->getArgumentAsVarAtIndex(i)->getIsWeight()) {
          os << llvm::formatv("  {0}_node->arg()->is_weight = true;\n",
                              newNodeName);
        }
        os << llvm::formatv("  {0}_node->arg()->type = LiteType::GetTensorTy(TARGET(kUnk), PRECISION(kFloat), DATALAYOUT(kUnk));\n",
                            newNodeName);
        os << llvm::formatv("  auto* {0}_t = scope->NewTensor(\"{1}\");\n",
                            newNodeName, newNodeName);
        os << llvm::formatv("  {0}_t->set_precision(paddle::lite_api::PrecisionType::kFloat);\n",
                            newNodeName);

        os << llvm::formatv("  {0}_t->CopyDataFrom({1});\n",
                            newNodeName, innerOpKey + "_val");
        os << llvm::formatv(
            "  {0}_desc.SetInput(\"{1}\", {{\"{2}\"});\n",
            targetPatOpKey, targetPatArgSlotNames[i], newNodeName);

        IR_NODE_LINK_TO_code << llvm::formatv(
                                    "  IR_NODE_LINK_TO({0}_node, {1}_op_node);\n",
                                    newNodeName, targetPatOpKey).str();
      }
    }
  }

  for (unsigned i = 0, c = targetPatOp->getResNames().size(); i < c; i++) {
    os << llvm::formatv(
        "  {0}_desc.SetOutput(\"{1}\", {matched.at(\"{2}\")->arg()->name});\n",
        targetPatOpKey,
        targetPatOp->getResNames()[i], srcPatOutputNames[i]);
  }


  // handle AttrToSet
  auto &attrsToSet = pat.getAttrsToSet();
  if (attrsToSet.count(targetPatOp->getKey())) {
    for (auto &attr : attrsToSet.at(targetPatOp->getKey())) {
      genSetAttrCode(targetPatOpKey + "_desc", attr, os);
    }
  }

  //handle attrToMap and attrToSet(by attr as argument)
  unsigned targetPatArgCount = targetPatArgNames.size();
  auto &targetPatOpArgs = targetPatOp->getArguments();

  //DirectCompute不需要设置attr
  unsigned size = targetPatOp->getType() == "DirectCompute" ?
                                            0 : targetPatOpArgs.size();
  for (unsigned  i = 0; i < size; i++) {
    if (targetPatOp->getArgumentTypeAtIndex(i) != TDOpArgument::attribute)
      continue;

    AttrToSet attr;
    attr.attrName = targetPatArgSlotNames[i];
    attr.dataType = targetPatOp->getArgumentAsAttrAtIndex(i)->getDataType();

    if (i < targetPatArgCount) {
      //假如有绑定且绑定在上下文中找得到，按map处理
      std::string actualArgName = targetPatArgNames[i];
      if (actualArgName != ""
          && argKeyToSrcPatOpKey.count(actualArgName) != 0) {
        std::string thatOpKey = argKeyToSrcPatOpKey[actualArgName];
        auto thatOp = key2op[thatOpKey];

        std::string thatAttrName
            = thatOp->getArgSlotNameByActualArgName(actualArgName);
        std::string thisAttrName = attr.attrName;
        std::string dataType = attr.dataType;

        os << llvm::formatv(
            "  {0}_desc.SetAttr(\"{1}\", "
            "matched.at(\"{2}\")->stmt()->op_info()->GetAttr<{3}>(\"{4}\"));\n",
            targetPatOpKey, thisAttrName, thatOpKey, dataType, thatAttrName);
        continue;
      }
      //否则，如果有值，采用参数值
      else if (static_cast<TDPatternAttrNode*>(
                   targetPattern->getArguments()[i].get())
                   ->getAttr() != nullptr) {
        attr.value = static_cast<TDPatternAttrNode*>(
                         targetPattern->getArguments()[i].get())
                         ->getAttr()->getValue();
      }
      //否则，使用默认值
      else {
        attr.value = targetPattern->getSetOrDefaultAttrAtIndex(i)->getValue();
      }
    }
    //实际参数个数少于形式参数个数，少的那部分必须用默认值
    else {
      attr.value = targetPatOp->getArgumentAsAttrAtIndex(i)->getValue();
    }
    genSetAttrCode(targetPatOpKey + "_desc", attr, os);
  }

  // handle AttrToCopy
  auto &attrsToCp = pat.getAttrsToCopy();
  for (auto &attr : attrsToCp) {
    if (attr.attrName.at(0) == '#')
      continue;
    std::string fromOpKeyed = opDesignatedKey2opKey.at(attr.from);
    std::string toOpKeyed = opDesignatedKey2opKey.at(attr.to);
    os << llvm::formatv(
        "  op_desc.SetAttr(\"{0}\", "
        "matched.at(\"{1}\")->stmt()->op_info()->GetAttr<{2}>(\"{3}\"));\n",
        attr.attrName, fromOpKeyed, attr.dataType, attr.attrName);
  }

  if (needCopyInputScale) {
    os << SetInputScaleStr.str();
  }

  os << llvm::formatv(
      "  auto {0}_op = LiteOpRegistry::Global().Create(\"{1}\");\n",
      targetPatOpKey, targetPatOpType);
  os << llvm::formatv("  {0}_op->Attach({1}_desc, scope);\n",
                      targetPatOpKey, targetPatOpKey);

  os << llvm::formatv("  auto* {0}_op_node = "
                      "graph->GraphCreateInstructNode({1}_op, valid_places);\n",
                      targetPatOpKey, targetPatOpKey);

  os << IR_NODE_LINK_TO_code.str();
  //TODO: 这里只需要连接到第一个输出节点么?
  if (isRoot) {
    os << llvm::formatv("  IR_NODE_LINK_TO({0}_op_node, matched.at(\"{1}\"));\n",
                        targetPatOpKey, srcPatOutputNames[0]);
  }
  else {
  }

  return targetPatOpKey;
}


void PdLiteGraphOptPassEmitter::handleTargetPatTopological(
                                          PdGraphOpt::TDPattern &pat,
                                          raw_ostream &os) {
  const auto &topo = pat.getTargetTopological();
  for (unsigned i = 0; i < topo.size(); i++) {
    TDPatternNode *node = topo[i];
    if (node->getNodeType() == TDPatternNode::Op) {
      auto opNode = static_cast<TDPatternOpNode*>(node);

      if (opNode->getOp()->getType() == "DirectCompute") {
        handleDirectComputeV2(opNode, pat, os);
      }
      else {
        if (i != topo.size() - 1)
          llvm::PrintFatalError("multiple none direct compute op in "
                                "target pattern has not been supported now");
        else {
          dfsResPatDagV2(opNode, true, os, pat);
        }
      }
    }
    //不需要处理var node或者attr node
    else {
    }
  }
}

std::pair<std::string, std::string>
PdLiteGraphOptPassEmitter::handleDirectComputeV2(TDPatternOpNode *dag,
                                                 TDPattern &pat,
                                               raw_ostream& os) {

  std::string computeType = dag->getOp()->getDirectComputeType();
  std::string computeKey = registerOp(dag);
  if (dag->getDesignatedOutputKey() != "") {
    computeKey = dag->getDesignatedOutputKey();
  }

  //本directCompute节点调用计算函数时传入的实参名称
  std::vector<std::string> computeVarSymbols;
  std::vector<bool> computeVarSymbolIsPointer;
  std::string dataType = "float";

  //实参个数和形参个数是否匹配
  if (dag->getArgNames().size() != dag->getOp()->getArgNames().size()) {
    PrintFatalError("DirectCompute expects 2 arg but get "
                    + std::to_string(dag->getArgNames().size()));
  }

  for(unsigned i = 0; i < dag->getArgNames().size(); i++) {
    auto node = dag->getArguments()[i].get();

    //如果是另一个directCompute
    //因为按拓扑顺序遍历，这个directCompute已经生成过了
    //这里只需要确定它生成的变量叫什么名字
    if (node->getNodeType() == PdGraphOpt::TDPatternNode::Op) {
      auto opNode = static_cast<TDPatternOpNode*>(node);
      if (opNode->getDesignatedOutputKey() != "") {
        computeVarSymbols.push_back(opNode->getDesignatedOutputKey() + "_val");
      }
      else {
        computeVarSymbols.push_back(op2key[opNode] + "_val");
      }
      computeVarSymbolIsPointer.push_back(false);
    }

    else if (node->getNodeType() == PdGraphOpt::TDPatternNode::Var) {
      ArgumentKeySourceInfo argInfo;
      //如果是绑定到sourcePattern中的实参，则从那里获取计算需要的数据
      if (getKeySourceInfoByKey(dag->getArgNames()[i], argInfo)) {
        if (argInfo.dataType != dataType) {
          PrintFatalError("DirectCompute args type mismatch");
        }
        if (dataType != "float") {
          PrintFatalError("DirectCompute only support float data type");
        }

        std::string varName = argInfo.opKey + "_" + argInfo.key + "_val";
        //如果绑定到的是一个variable（tensor）
        if (argInfo.argType == PdGraphOpt::TDOpArgument::variable) {
          os << llvm::formatv(
              "  auto {0} = scope->FindVar(matched.at(\"{1}\")->arg()"
              "->name)->GetMutable<lite::Tensor>();\n",
              varName, argInfo.key);
        }
        //如果绑定到的是一个attribute
        else {
          os << llvm::formatv("  auto {0} = "
                              "matched.at(\"{1}\")->stmt()->op_info()->GetAttr<{"
                              "2}>(\"{3}\");\n",
                              varName, argInfo.opKey, argInfo.dataType,
                              argInfo.slotName);
        }
        computeVarSymbols.push_back(varName);
        computeVarSymbolIsPointer.push_back(true);
      }
      //如果是绑定到另一个directCompute的结果，则
      else if (pat.getTargetOpByDesignatedKey(dag->getArgNames()[i])) {
        computeVarSymbols.push_back(dag->getArgNames()[i] + "_val");
        computeVarSymbolIsPointer.push_back(false);
      }
      else {
        llvm::PrintFatalError("Can't resolve argument of directCompute");
      }
    }
  }

  static std::unordered_map<std::string, std::string> computeType2FuncName {
      {"DirectEleWiseAdd", "directEleWiseAddFloat32"},
      {"DirectEleWiseSub", "directEleWiseSubFloat32"},
      {"DirectEleWiseMul", "directEleWiseMulFloat32"},
      {"DirectEleWiseDiv", "directEleWiseDivFloat32"},
  };

  if (computeType2FuncName.count(computeType)) {
    std::string& funcName = computeType2FuncName[computeType];
    os << llvm::formatv("  auto {0}_val = {1}({2}{3}, {4}{5});\n",
                        computeKey,
                        funcName,
                        computeVarSymbolIsPointer[0] ? "" : "&",
                        computeVarSymbols[0],
                        computeVarSymbolIsPointer[1] ? "" : "&",
                        computeVarSymbols[1]);
  }
  else if (computeType == "DirectEleWiseSqrt") {
    os << llvm::formatv("  auto {0}_val = directEleWiseSqrtFloat32({1}{2});\n",
                        computeKey,
                        computeVarSymbolIsPointer[0] ? "" : "&",
                        computeVarSymbols[0]);
  }
  else if (computeType == "DirectRowWiseMul") {
    //读取x_num_col_dims的属性值
    auto attr = dag->getSetOrDefaultAttrAtIndex(2);
    os << llvm::formatv("  auto {0}_val = directRowWiseMulFloat32({1}{2}, {3}{4}, {5});\n",
                        computeKey,
                        computeVarSymbolIsPointer[0] ? "" : "&",
                        computeVarSymbols[0],
                        computeVarSymbolIsPointer[1] ? "" : "&",
                        computeVarSymbols[1],
                        attr->getValue());
  }

  return std::make_pair(computeKey, dataType);
}

bool
PdLiteGraphOptPassEmitter::getKeySourceInfoByKey(std::string key,
                                                 ArgumentKeySourceInfo &info) {
  if (argKeyToSrcPatOpKey.count(key) == 0) {
    return false;
  }
  auto theOpKey = argKeyToSrcPatOpKey[key];
  auto theOp = key2op[theOpKey];
  long index = theOp->getIndexByActualArgName(key);
  auto argType = theOp->getOp()->getArgumentTypeAtIndex(index);

  info.key = key;
  info.index = index;
  info.opKey = theOpKey;
  info.argType = argType;
  info.slotName = theOp->getOp()->getArgNames()[index];
  if (argType == PdGraphOpt::TDOpArgument::variable) {
    info.dataType = theOp->getOp()->getArgumentAsVarAtIndex(index)->getDataType();
  }
  else {
    info.dataType = theOp->getOp()->getArgumentAsAttrAtIndex(index)->getDataType();
  }

  return true;
}

void PdLiteGraphOptPassEmitter::EmitFuserHeader(PdGraphOpt::TDPattern &pat,
                                                raw_ostream &os) {

  std::string patName = pat.getNameWithoutPat();
  const auto &variableOpTypes = pat.getVariableOpTypes();
  os << "class " << patName << "Fuser"
     << " : public FuseBase {\n";
  os << "public:\n";

  // 生成这样的构造函数：
  // explicit MatchMatrixActFuser(std::string activation)
  //             : activation_(activation) {}
  if (variableOpTypes.size() > 0) {
    os << "  explicit " << patName << "Fuser(";
    bool first = true;
    std::stringstream assignStr;
    for (auto &item : variableOpTypes) {
      if (first) {
        first = false;
      } else {
        os << ", ";
        assignStr << ", ";
      }
      assignStr << llvm::formatv("{0}({1})", item, item + "Arg").str();
      os << "std::string"
         << " " << item << "Arg";
    }
    os << "): ";
    os << assignStr.str() << "{}\n";
  }

  os << "  void BuildPattern() override;\n";
  os << "  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) "
        "override;\n";
  os << "private:\n";
  for (auto &item : variableOpTypes) {
    os << "  std::string"
       << " " << item << ";\n";
  }

  os << "};\n";

  os << "\n";
}

void PdLiteGraphOptPassEmitter::EmitFuserImpl(PdGraphOpt::TDPattern &pat,
                                              raw_ostream &os) {
  EmitBuildPatternMethod(pat, os);
  os << "\n";
  EmitBuildNewGraphMethod(pat, os);
}

void PdLiteGraphOptPassEmitter::EmitPassHeader(PdGraphOpt::TDPattern &pat,
                                               raw_ostream &os) {

  os << llvm::formatv("class {0}FusePass : public ProgramPass {{\n",
                      pat.getNameWithoutPat());
  os << R"(public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};
)";

}

void PdLiteGraphOptPassEmitter::EmitPassImpl(PdGraphOpt::TDPattern &pat,
                                             raw_ostream &os) {

  os << llvm::formatv("void {0}FusePass::Apply(\n",
                      pat.getNameWithoutPat());
  os << "    const std::unique_ptr<SSAGraph>& graph) {\n";
  os << llvm::formatv("  fusion::{0}Fuser fuser;\n",
                      pat.getNameWithoutPat());
  os << "  fuser(graph.get());\n";
  os << "}\n";
}

void PdLiteGraphOptPassEmitter::EmitRegisterPass(PdGraphOpt::TDPattern &pat,
                                             raw_ostream &os) {

  os << "REGISTER_MIR_PASS(lite_"
     << convertCamelToSnake(pat.getNameWithoutPat() + "FusePass")
     << ",\n";
  os << "                  paddle::lite::mir::"
     << pat.getNameWithoutPat() << "FusePass" << ")\n";
  for (auto &target : pat.getBindTargets()) {
    os << llvm::formatv("    .BindTargets({{TARGET({0})})\n",
                        target);
  }
  for (auto &exclude : pat.getExcludeTargets()) {
    os << llvm::formatv("    .ExcludeTargets({{TARGET({0})})\n",
                        exclude);
  }
  if (pat.getKernelName() != "") {
    os << llvm::formatv("    .BindKernel(\"{0}\")\n", pat.getKernelName());
  }
  os << ";\n";
}


void PdLiteGraphOptPassEmitter::EmitSingleFileHeader(raw_ostream &os) {
  os << license;
  os << "\n";
  os << "#pragma once";
  os << "\n\n";

  os << singleFileInclude;
  os << "\n\n";
}

void PdLiteGraphOptPassEmitter::run(raw_ostream &OS) {

  PdGraphOpt::RecordConverter converter(Records);
  if (singleFileMode) {
    EmitSingleFileHeader(OS);
  }
  OS << nameSpaceFusionBegin;
  OS << "\n";
  OS << directElementWiseCompute;
  OS << "\n";
  OS << nameSpaceFusionEnd;

  for (PdGraphOpt::TDPattern &pat : converter.getPatterns()) {
    OS << "//==============================================================================\n";
    OS << "//" << pat.getName() << "\n";
    OS << "//==============================================================================\n";
    OS << nameSpaceFusionBegin;
    OS << "\n";
    EmitFuserHeader(pat, OS);
    OS << "\n";
    EmitFuserImpl(pat, OS);
    OS << "\n";
    OS << nameSpaceFusionEnd;

    OS << "\n";

    OS << nameSpaceMirBegin;
    OS << "\n";
    EmitPassHeader(pat, OS);
    OS << "\n";
    EmitPassImpl(pat, OS);
    OS << "\n";
    OS << nameSpaceMirEnd;
    OS << "\n";
    EmitRegisterPass(pat, OS);

    resetState();
  }
}

namespace llvm {

void EmitPaddleLiteGraphOptPass(RecordKeeper &RK, raw_ostream &OS) {
  PdLiteGraphOptPassEmitter(RK).run(OS);
}

} // namespace llvm
