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
}

namespace {

//目前主要生成Fuser中的三个方法。
class PdLiteGraphOptPassEmitter {

private:
  RecordKeeper &Records;

  //是否输出为单文件
  bool singleFileMode{true};

  //把op按类型计数。用于在给op分配key时生成唯一字符串。
  std::map<std::string, unsigned> opKeyAllocationRecord;

  //把var按类型计数。
  std::map<std::string, unsigned> varKeyAllocationRecord;

  // op到它被分配的key间的映射。
  std::map<PdGraphOpt::TDPatternOpNode *, std::string> op2key;

  // op的key到op间的映射。
  std::map<std::string, PdGraphOpt::TDPatternOpNode *> key2op;

  //用户给op设定的key（如果有的话）到op被分配的key间的映射。
  std::map<std::string, std::string> opDesignatedKey2opKey;

  // argument的key到其所属的op的映射。
  std::map<std::string, std::string> argKeyToSrcPatOpKey;

  std::vector<std::string> srcPatOutputSlotNames{"Out"};

  /// Pattern（有向图）的首个节点，融合时会作为替换点位
  PdGraphOpt::TDPatternOpNode *leadingOpNode{nullptr};

  /// 生成了一个pass之后，重置状态
  void resetState() {
    leadingOpNode = nullptr;
    srcPatOutputSlotNames = {"Out"};
    argKeyToSrcPatOpKey.clear();
    opDesignatedKey2opKey.clear();
    opKeyAllocationRecord.clear();
    varKeyAllocationRecord.clear();
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

  void EmitInsertNewNodeMethod(PdGraphOpt::TDPattern &pat, raw_ostream &os);

  void EmitGenOpDescMethod(PdGraphOpt::TDPattern &pat, raw_ostream &os);

  void EmitPassHeader(PdGraphOpt::TDPattern &pat, raw_ostream &os);

  void EmitPassImpl(PdGraphOpt::TDPattern &pat, raw_ostream &os);

  std::string dfsSrcPatDag(PdGraphOpt::TDPatternOpNode *dag, raw_ostream &os,
                        PdGraphOpt::TDPattern &pat);

  std::string dfsResPatDag(PdGraphOpt::TDPatternOpNode *dag, raw_ostream &os,
                           PdGraphOpt::TDPattern &pat);

  std::string registerOp(PdGraphOpt::TDPatternOpNode *opNode);

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

} // anonymous namespace

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
  // TODO: string 要不要加std sting？
  else {
    os << llvm::formatv(
        "  {0}->assert_op_attr<{1}>(\"{2}\", {3});\n", opKey, attr.dataType,
        attr.attrName,
        attr.dataType != "string" ? attr.value : "\"" + attr.value + "\"");
  }
}

//遍历sourcePattern，生成定义VarNode、OpNode及其定义它们拓扑关系的代码
//在EmitBuildPatternMethod中被调用
std::string
PdLiteGraphOptPassEmitter::dfsSrcPatDag(PdGraphOpt::TDPatternOpNode *dag,
                                     raw_ostream &os,
                                     PdGraphOpt::TDPattern &pat) {

  auto op = dag->getOp();
  std::string opType = op->getTypeAuto();

  std::string opKey = registerOp(dag);
  // TODO: 处理op的输出被绑定了key以及op多输出的情况

  //生成该dag的op对应的OpNode
  os << "  //Start op " << opKey << "\n";
  os << llvm::formatv("  auto* {0} = OpNode(\"{1}\", {2});", opKey, opKey,
                      opType);
  os << "\n";
  // TODO: 什么时候它不是Intermediate？
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
    if (nodeType == PdGraphOpt::TDPatternNode::Op) {
      patWithoutDag = false;
    }
    if (nodeType == PdGraphOpt::TDPatternNode::Attr) {
      TDPatternAttrNode* node = (TDPatternAttrNode*)arg;
      if (node->getAttr() != nullptr) {
        AttrToAssert newAssert;
        newAssert.target = op->getKey();
        newAssert.value = node->getAttr()->getValue();
        newAssert.dataType = node->getAttr()->getType();
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
    if (args[i]->getNodeType() == PdGraphOpt::TDPatternNode::Var) {
      //这个Var的名称（或者说是key，dag中指定的），作为生成VarNode时的变量名
      std::string argKey = argNames[i];
      inputs.push_back(argKey);
      argKeyToSrcPatOpKey[argKey] = opKey;
      os << llvm::formatv(
          "  auto* {0} = VarNode(\"{1}\")->assert_is_op_input({2}, "
          "\"{3}\")",
          argKey, argKey, opType, opArgNames[i]);

      auto varArgPtr =
          static_cast<PdGraphOpt::TDPatternVarNode *>(args[i].get());
      if (varArgPtr->getVar() && varArgPtr->getVar()->getIsWeight()) {
        os << "->assert_is_persistable_var()";
      }
      os << ";\n";
    }
    // TODO: 增加对var的约束机制，比如维度的约束
    else if (args[i]->getNodeType() == PdGraphOpt::TDPatternNode::Attr) {
      std::string argKey = argNames[i];
      inputs.push_back(argKey);
      argKeyToSrcPatOpKey[argKey] = opKey;
    }

    //如果这个dag的参数是DagInit，也就是说它是个Dag，所以递归调用本方法。
    else if (args[i]->getNodeType() == PdGraphOpt::TDPatternNode::Op) {

      auto opArgPtr = static_cast<PdGraphOpt::TDPatternOpNode *>(args[i].get());
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
//生成源码中`GenOpDesc`这个方法。
void PdLiteGraphOptPassEmitter::EmitGenOpDescMethod(PdGraphOpt::TDPattern &pat,
                                                    raw_ostream &os) {
  assert(leadingOpNode != nullptr && "leadingOpNode not found.");

  os << "cpp::OpDesc " << pat.getNameWithoutPat()
     << "Fuser::GenOpDesc(const key2nodes_t& matched) {\n";

  std::string patHeadOpKey = op2key.at(leadingOpNode);
  os << llvm::formatv(
      "  auto op_desc = *matched.at(\"{0}\")->stmt()->op_info();\n",
      patHeadOpKey);

  //开始处理SetInputScale的部分。
  //由于SetInputScale的代码有分开的两部分，这里先把后面那部分一并生成。
  std::stringstream SetInputScaleStr;
  SetInputScaleStr << "  if (is_quantized_op) {\n";
  std::stringstream GetInputScaleStr;
  GetInputScaleStr << "  if (is_quantized_op) {\n";

  if (pat.getNeedCopyInputScale()) {
    os << "  bool is_quantized_op = true;\n";
    for (auto &&attr : pat.getAttrsToCopy()) {
      if (attr.attrName != "#INPUT_SCALE")
        continue;

      std::string formArgName =
          key2op[argKeyToSrcPatOpKey[attr.from]]->getArgSlotNameByActualArgName(
              attr.from);
      os << llvm::formatv(
          "  auto input_{0}_name = op_desc.Input(\"{1}\").front();\n",
          attr.from, formArgName);
      os << llvm::formatv("  std::vector<float> {0}_scale_vct;\n", attr.from);

      os << llvm::formatv(
          "  is_quantized_op &= op_desc.HasInputScale(input_{0}_name);\n",
          attr.from);

      GetInputScaleStr
          << llvm::formatv(
                 "    {0}_scale_vct = op_desc.GetInputScale(input_{1}_name);\n",
                 attr.from, attr.from)
                 .str();

      SetInputScaleStr
          << llvm::formatv(
                 "    op_desc.SetInputScale(matched.at(\"{0}\")->arg()->name, "
                 "{1}_scale_vct);\n",
                 attr.from, attr.from)
                 .str();
    }
    SetInputScaleStr << "  }\n";
    GetInputScaleStr << "  }\n";

    os << GetInputScaleStr.str();
  } //结束处理SetInputScale的部分。

  os << "  op_desc.mutable_inputs()->clear();\n";
  os << "  op_desc.mutable_outputs()->clear();\n";

  auto targetPattern = pat.getTargetPatternRoot();
  auto targetPatOp = targetPattern->getOp();
  auto &targetPatArgSlotNames = targetPatOp->getArgNames();
  auto &targetPatArgNames = targetPattern->getArgNames();

  os << llvm::formatv("  op_desc.SetType(\"{0}\");\n", targetPatOp->getType());

  for (unsigned i = 0, count = targetPatArgSlotNames.size(); i < count; i++) {
    if (targetPatOp->getArgumentTypeAtIndex(i)
        != PdGraphOpt::TDOpArgument::variable) {
      continue;
    }
    os << llvm::formatv(
        "  op_desc.SetInput(\"{0}\", {matched.at(\"{1}\")->arg()->name});\n",
        targetPatArgSlotNames[i], targetPatArgNames[i]);
  }

    for (unsigned i = 0, c = targetPatOp->getResNames().size(); i < c; i++) {
      os << llvm::formatv(
          "  op_desc.SetOutput(\"{0}\", {matched.at(\"{1}\")->arg()->name});\n",
          targetPatOp->getResNames()[i],
          srcPatOutputSlotNames[i]);
    }

//  os << llvm::formatv(
//      "  op_desc.SetOutput(\"{0}\", {matched.at(\"{1}\")->arg()->name});\n",
//      targetPatOp->getResNames()[0], srcPatOutputSlotNames);

  // handle AttrToSet
  //这里并没有考虑AttrToSet指定的target，目前认为target就是融合后的那个结点
  auto &attrsToSet = pat.getAttrsToSet();
  if (attrsToSet.count(targetPatOp->getKey())) {
    for (auto &attr : attrsToSet.at(targetPatOp->getKey())) {
      genSetAttrCode("op_desc", attr, os);
    }
  }

  //handle attrToMap and attrToSet(by attr as argument)
  unsigned targetPatArgCount = targetPatArgNames.size();
  auto &targetPatOpArgs = targetPatOp->getArguments();
  for (unsigned  i = 0; i < targetPatOpArgs.size(); i++) {
    if (targetPatOp->getArgumentTypeAtIndex(i)
        != PdGraphOpt::TDOpArgument::attribute)
      continue;

    AttrToSet attr;
    attr.attrName = targetPatArgSlotNames[i];
    attr.dataType = targetPatOp->getArgumentAsAttrAtIndex(i)->getType();

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
            "  op_desc.SetAttr(\"{0}\", "
            "matched.at(\"{1}\")->stmt()->op_info()->GetAttr<{2}>(\"{3}\"));\n",
            thisAttrName, thatOpKey, dataType, thatAttrName);
        continue;
      }
      //否则，如果有值，采用参数值
      else if (static_cast<TDPatternAttrNode*>(
                   targetPattern->getArguments()[i].get())
                   ->getAttr() != nullptr) {
        attr.value = static_cast<TDPatternAttrNode*>(
                        targetPattern->getArguments()[i].get())->getAttr()->getValue();
      }
      //否则，使用默认值
      else {
        attr.value = targetPatOp->getArgumentAsAttrAtIndex(i)->getValue();
      }
    }
    //这个必然使用默认值
    else {
      attr.value = targetPatOp->getArgumentAsAttrAtIndex(i)->getValue();
    }
    genSetAttrCode("op_desc", attr, os);
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

  if (pat.getNeedCopyInputScale()) {
    os << SetInputScaleStr.str();
  }

  os << "  return op_desc;\n";
  os << "}\n";
}

std::string PdLiteGraphOptPassEmitter::dfsResPatDag(
                              PdGraphOpt::TDPatternOpNode *dag,
                              raw_ostream &os,
                              PdGraphOpt::TDPattern &pat) {
  return "";
}

//生成源码中`InsertNewNode`这个方法。
void PdLiteGraphOptPassEmitter::EmitInsertNewNodeMethod(
    PdGraphOpt::TDPattern &pat, raw_ostream &os) {
  assert(leadingOpNode != nullptr && "leadingOpNode not found.");

  std::string leadingOpType = leadingOpNode->getOp()->getType();
  std::string leadingOpKey = op2key.at(leadingOpNode);

  auto targetPat = pat.getTargetPatternRoot();
  std::string targetPatOpType = targetPat->getOp()->getType();
  std::string targetPatOpKey = registerOp(targetPat);

  os << "void " << pat.getNameWithoutPat()
     << "Fuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {\n";

  os << "  auto op_desc = GenOpDesc(matched);\n";

  os << llvm::formatv(
      "  auto {0}_op = LiteOpRegistry::Global().Create(\"{1}\");\n",
      targetPatOpKey, targetPatOpType);

  os << llvm::formatv("  auto {0} = matched.at(\"{1}\")->stmt()->op();\n",
                      leadingOpKey, leadingOpKey);

  os << llvm::formatv("  auto* scope = {0}->scope();\n", leadingOpKey);
  os << llvm::formatv("  auto& valid_places = {0}->valid_places();\n",
                      leadingOpKey);
  os << llvm::formatv("  {0}_op->Attach(op_desc, scope);\n", targetPatOpKey);

  os << llvm::formatv("  auto* new_op_node = "
                      "graph->GraphCreateInstructNode({0}_op, valid_places);\n",
                      targetPatOpKey);

  for (unsigned i = 0; i < targetPat->getArgNames().size(); i++) {
    if (targetPat->getOp()->getArgumentTypeAtIndex(i)
        == PdGraphOpt::TDOpArgument::variable) {
      os << llvm::formatv(
          "  IR_NODE_LINK_TO(matched.at(\"{0}\"), new_op_node);\n",
          targetPat->getArgNames()[i]);
    }

  }
  //TODO: 这里只需要连接到第一个输出节点么？
  os << llvm::formatv("  IR_NODE_LINK_TO(new_op_node, matched.at(\"{0}\"));\n",
                      srcPatOutputSlotNames[0]);
  os << "}\n";
}

//生成源码中`BuildPattern`这个方法。这个方法是生成代码时最先被调用的。
void PdLiteGraphOptPassEmitter::EmitBuildPatternMethod(
    PdGraphOpt::TDPattern &pat, raw_ostream &os) {
  auto *srcPatRoot = pat.getSourcePatternRoot();

  os << "void " << pat.getNameWithoutPat() << "Fuser::BuildPattern() {\n";

  std::string innerOpKey = dfsSrcPatDag(srcPatRoot, os, pat);
  this->srcPatOutputSlotNames = srcPatRoot->getOp()->getResNames();
  std::string innerOpType = srcPatRoot->getOp()->getTypeAuto();

  for (auto& srcPatOutputSlotName: this->srcPatOutputSlotNames) {
    os << llvm::formatv(
        "  auto* {0} = VarNode(\"{1}\")->assert_is_op_output({2}, \"{3}\");\n",
        srcPatOutputSlotName, srcPatOutputSlotName, innerOpType,
        srcPatOutputSlotName);
  }
  os << llvm::formatv("  std::vector<PMNode*> {0}_outputs{1};\n",
                      innerOpKey,
                      genListString(this->srcPatOutputSlotNames,",", "{", "}"));
  os << llvm::formatv("  {0}_inputs >> *{1} >> {0}_outputs;\n", innerOpKey,
                      innerOpKey, innerOpKey);

  os << "}\n";
}

void PdLiteGraphOptPassEmitter::EmitFuserHeader(PdGraphOpt::TDPattern &pat,
                                                raw_ostream &os) {
  if (!singleFileMode) {
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
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"
#include <string>
)";
    os << "\n";
  }


  os << R"(namespace paddle {
namespace lite {
namespace mir {
namespace fusion {
)";

  std::string patName = pat.getNameWithoutPat();
  const auto &variableOpTypes = pat.getVariableOpTypes();
  os << "\n";
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
  os << "  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;\n";
  for (auto &item : variableOpTypes) {
    os << "  std::string"
       << " " << item << ";\n";
  }

  os << "};\n";

  os << "\n";
  os << R"(}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
)";
}

void PdLiteGraphOptPassEmitter::EmitFuserImpl(PdGraphOpt::TDPattern &pat,
                                              raw_ostream &os) {
  if (!singleFileMode) {

  }
  os << R"(namespace paddle {
namespace lite {
namespace mir {
namespace fusion {
)";
  EmitBuildPatternMethod(pat, os);
  os << "\n";
  EmitInsertNewNodeMethod(pat, os);
  os << "\n";
  EmitGenOpDescMethod(pat, os);
  os << "\n";
  os << R"(}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
)";
}

void PdLiteGraphOptPassEmitter::EmitPassHeader(PdGraphOpt::TDPattern &pat,
                                               raw_ostream &os) {
  if (!singleFileMode) {
    os << R"(#pragma once

#include "lite/core/optimizer/mir/pass.h"
#include <memory>
#include <string>
)";
  }
  os << R"(
namespace paddle {
namespace lite {
namespace mir {
)";
  os << llvm::formatv("class {0}FusePass : public ProgramPass {{\n",
                      pat.getNameWithoutPat());
  os << R"(public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};
)";
  os << R"(}  // namespace mir
}  // namespace lite
}  // namespace paddle
)";
}

void PdLiteGraphOptPassEmitter::EmitPassImpl(PdGraphOpt::TDPattern &pat,
                                             raw_ostream &os) {
  os << R"(namespace paddle {
namespace lite {
namespace mir {
)";
  os << llvm::formatv("void {0}FusePass::Apply(\n",
                      pat.getNameWithoutPat());
  os << "    const std::unique_ptr<SSAGraph>& graph) {\n";
  os << llvm::formatv("  fusion::{0}Fuser fuser;\n",
                      pat.getNameWithoutPat());
  os << "  fuser(graph.get());\n";
  os << "}\n\n";

  os << R"(}  // namespace mir
}  // namespace lite
}  // namespace paddle
)";

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
  os << llvm::formatv("    .BindKernel(\"{0}\");\n", pat.getKernelName());
}

void PdLiteGraphOptPassEmitter::EmitSingleFileHeader(raw_ostream &os) {
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
#include <vector>
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pass.h"

)";
  os << "\n";
}

void PdLiteGraphOptPassEmitter::run(raw_ostream &OS) {

  PdGraphOpt::RecordConverter converter(Records);
  if (singleFileMode) {
    EmitSingleFileHeader(OS);
  }
  for (PdGraphOpt::TDPattern &pat : converter.getPatterns()) {
    OS << "//===============================================================\n";
    OS << "//" << pat.getName() << "\n";
    OS << "//===============================================================\n";
    EmitFuserHeader(pat, OS);
    OS << "\n";
    EmitFuserImpl(pat, OS);
    OS << "\n";
    EmitPassHeader(pat, OS);
    OS << "\n";
    EmitPassImpl(pat, OS);
    OS << "\n";
    resetState();
  }
}

namespace llvm {

void EmitPaddleLiteGraphOptPass(RecordKeeper &RK, raw_ostream &OS) {
  PdLiteGraphOptPassEmitter(RK).run(OS);
}

} // namespace llvm
