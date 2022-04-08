//===- PdLiteGraphOptPassEmitter.cpp -                          -*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// This Tablegen backend emits Paddle Lite fusion passes code.
//
//===----------------------------------------------------------------------===//

#include "PdLiteGraphOpt/RecordConverter.h"
#include "PdLiteGraphOpt/CodeFragment.h"
#include "PdLiteGraphOpt/helper.h"
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

// 生成fusion pass
class PdLiteGraphOptPassEmitter {

private:
  RecordKeeper &Records;
  raw_ostream& os;

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

  //source pattern输出结果的变量符号名
  std::vector<std::string> srcPatOutputNames{"Out"};

  // Pattern（有向图）的首个节点，融合时会作为替换点位
  PdGraphOpt::TDPatternOpNode *leadingOpNode{nullptr};

  //
  std::unordered_set<std::string> computeVarSymbolPool;

  /// 生成了一个pass之后，重置状态
  void resetState() {
    leadingOpNode = nullptr;
    srcPatOutputNames = {"Out"};
    argKeyToSrcPatOpKey.clear();
    opDesignatedKey2opKey.clear();
    opKeyAllocationRecord.clear();
    computeVarSymbolPool.clear();
    key2op.clear();
    op2key.clear();
  }

  void EmitSingleFileHeader();

  void EmitFuserHeader(TDPattern &pat);

  void EmitFuserImpl(TDPattern &pat);

  void EmitBuildPatternMethod(TDPattern &pat);

  void EmitBuildNewGraphMethod(TDPattern &pat);

  void EmitPassHeader(TDPattern &pat);

  void EmitPassImpl(TDPattern &pat);

  void EmitRegisterPass(TDPattern &pat);

  std::string dfsSrcPatDag(TDPatternOpNode *dag, TDPattern &pat);

  std::string dfsResPatDag(TDPatternOpNode *dag, bool isRoot, TDPattern &pat);

  void handleTargetPatTopological(TDPattern &pat);

  std::pair<std::string, std::string>
    handleDirectCompute(TDPatternOpNode *dag, TDPattern &pat);

  std::string registerOp(TDPatternOpNode *opNode);

  bool getKeySourceInfoByKey(std::string key,ArgumentKeySourceInfo &info);

  std::string getNewOpKey(std::string opType);

public:
  PdLiteGraphOptPassEmitter(RecordKeeper &RK, raw_ostream& OS)
      :Records(RK), os(OS) {}

  void run(raw_ostream &OS);
};

} // PdGraphOpt namespace

std::string
PdLiteGraphOptPassEmitter::registerOp(TDPatternOpNode *opNode) {
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

/// 生成源码中的`BuildPattern`方法。这个方法是生成代码时最先被调用的。
void PdLiteGraphOptPassEmitter::EmitBuildPatternMethod(TDPattern &pat) {
  auto *srcPatRoot = pat.getSourcePatternRoot();

  os << "void " << pat.getNameWithoutPat() << "Fuser::BuildPattern() {\n";

  std::string innerOpKey = dfsSrcPatDag(srcPatRoot, pat);

  std::vector<std::string> srcPatOutputNames;
  for (auto &name : srcPatRoot->getOp()->getResNames()) {
    srcPatOutputNames.push_back(innerOpKey + "_" + name);
  }
  this->srcPatOutputNames = srcPatOutputNames;
  std::string innerOpType = srcPatRoot->getOp()->getTypeAuto();

  bool first = true;
  for (unsigned i = 0; i < srcPatOutputNames.size(); i++) {
    os << llvm::formatv(
        "  auto* {0} = VarNode(\"{1}\")->assert_is_op_output({2}, \"{3}\")",
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
PdLiteGraphOptPassEmitter::dfsSrcPatDag(TDPatternOpNode *dag, TDPattern &pat) {

  auto op = dag->getOp();
  std::string opType = op->getTypeAuto();
  std::string opKey = registerOp(dag);

  //这个op固有的参数名称列表，即这个op在td中定义时书写的参数名称列表
  auto slotArgNames = op->getArgNames();
  //这个op在pattern中的参数名称列表
  auto &argNames = dag->getArgNames();

  //生成该dag的op对应的OpNode
  os << "  //Start op " << opKey << "\n";
  os << llvm::formatv("  auto* {0} = OpNode(\"{1}\", {2});", opKey, opKey,
                      opType);
  os << "\n";
  //如果op没有被retain，设为AsIntermediate
  if (!(op->getKey() != "" && pat.isOpRetained(op->getKey()))) {
    os << "  " << opKey << "->AsIntermediate();\n";
  }

  //为这个OpNode添加断言（这种形式的断言现在不怎么用了）
  if (pat.getAttrsToAssert().count(op->getKey())) {
    auto &attrs = pat.getAttrsToAssert().at(op->getKey());
    for (auto &&attr : attrs) {
      genAttrAssertCode(attr, opKey, os);
    }
  }
  //为这个OpNode添加customTeller
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
        newAssert.attrName = slotArgNames[i];
        genAttrAssertCode(newAssert, opKey, os);
      }
    }
  }
  if (patWithoutDag) {
    this->leadingOpNode = dag;
  }

  auto &nodeCond = pat.getExtraCondByTargetKey(op->getKey());
  for (auto &&cond : nodeCond) {
    if (cond.conditionType == "InputRankAllEquals") {
      auto opVarNames = op->getVarNames();
      std::string rankEqStr = EmitInputDimChecker(
                                   opVarNames,
                                   std::vector<int>(opVarNames.size(), 0),
                                   std::vector<ComparisonOperator>());
      os << llvm::formatv("  {0}->assert_node_satisfied({1});\n", opKey,
                          rankEqStr);
    }
  }

  //这个op的参数名称列表，参数名用来作为VarNode的变量名。
  std::vector<std::string> inputs;

  //处理一个OpNode的参数，生成对应的VarNode
  for (unsigned i = 0, c = args.size(); i < c; i++) {
    //如果这个dag的参数是个Var，生成一个VarNode
    if (args[i]->getNodeType() == TDPatternNode::Var) {
      //这个Var的名称（或者说是key，dag中指定的实际参数名称），作为生成VarNode时的变量名
      std::string argKey = argNames[i];
      inputs.push_back(argKey);
      argKeyToSrcPatOpKey[argKey] = opKey;
      os << llvm::formatv(
          "  auto* {0} = VarNode(\"{1}\")->assert_is_op_input({2}, \"{3}\")",
          argKey, argKey, opType, slotArgNames[i]);

      auto varArgPtr = static_cast<TDPatternVarNode *>(args[i].get());
      if (varArgPtr->getVar() && varArgPtr->getVar()->getIsWeight()) {
        os << "->assert_is_persistable_var()";
      }
      if (!pat.isVarDirectlyUsedByTargetPattern(argKey)) {
        os << "->AsIntermediate()";
      }
      os << ";\n";

      //如果对这个var施加了一些extraAssertion
      auto &extraCond = pat.getExtraCondByTargetKey(argKey);
      for (auto &&cond : extraCond) {
        if (cond.conditionType == "RankEquals") {
          os << llvm::formatv(
              "  {0}->assert_node_satisfied([] (const Node* node) {\n"
              "    return assertRankEquals(node, \"{1}\", {2});\n  });\n",
              opKey, slotArgNames[i], cond.value1);
        }
        else if (cond.conditionType == "RankInRange") {
          os << llvm::formatv(
              "  {0}->assert_node_satisfied([] (const Node* node) {\n"
              "    return assertRankInRange(node, \"{1}\", {2}, {3});\n  });\n",
              opKey, slotArgNames[i], cond.value1, cond.value2);
        }
      }
    }
    //attribute已经处理过了
    else if (args[i]->getNodeType() == TDPatternNode::Attr) {
      std::string argKey = argNames[i];
      argKeyToSrcPatOpKey[argKey] = opKey;
    }
    //如果有嵌套的op（dag），递归处理
    // TODO: 处理op的输出被绑定了key以及op多输出的情况
    else if (args[i]->getNodeType() == TDPatternNode::Op) {

      auto opArgPtr = static_cast<TDPatternOpNode *>(args[i].get());
      std::string innerOpKey = dfsSrcPatDag(opArgPtr, pat);
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

/// 生成源码中的`InsertNewNode`方法
void PdLiteGraphOptPassEmitter::EmitBuildNewGraphMethod(TDPattern &pat) {
  assert(leadingOpNode != nullptr && "leadingOpNode not found.");
  std::string leadingOpKey = op2key.at(leadingOpNode);

  auto targetPat = pat.getTargetPatternRoot();

  os << "void " << pat.getNameWithoutPat()
     << "Fuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {\n\n";

  os << llvm::formatv("  auto* scope = matched.at(\"{0}\")->stmt()->op()->scope();\n",
                      leadingOpKey);
  os << llvm::formatv("  auto& valid_places = matched.at(\"{0}\")->stmt()->op()->valid_places();\n",
                      leadingOpKey);

  handleTargetPatTopological(pat);

  os << "}\n";
}

std::string PdLiteGraphOptPassEmitter::dfsResPatDag(
    TDPatternOpNode *dag,
    bool isRoot,
    TDPattern &pat) {

  TDOperator *op = dag->getOp();
  std::string opType = op->getType();
  std::string opKey = registerOp(dag);

  std::string retainKey = pat.getRetainKey(op->getKey());
  if (retainKey != "") {
    os << llvm::formatv("  OpInfo {0}_desc_main = "
                        "*matched.at(\"{1}\")->stmt()->op_info();\n",
                        opKey,
                        opDesignatedKey2opKey[retainKey]);
  }
  else {
    os << llvm::formatv("  OpInfo {0}_desc_main((cpp::OpDesc()));\n", opKey);
  }

  //开始处理SetInputScale的部分。
  //由于SetInputScale的代码有分开的两部分，这里先把后面那部分一并生成。
  std::stringstream SetInputScaleStr;
  std::stringstream SetInputScaleStrALL;
  std::stringstream GetInputScaleStr;

  // 判断是否需要拷贝InputScale
  bool needCopyInputScale = false;
  // TODO: 完善inputScale来自多个op的情况
  std::vector<AttrToCopy> inputScaleCopies;

  if (pat.getNeedCopyInputScale()) {
    for (auto &&attr : pat.getAttrsToCopy()) {
      if (attr.attrName == "#INPUT_SCALE"
          && attr.to == op->getKey()) {
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
          "  auto {0}_desc = *matched.at(\"{1}\")->stmt()->op_info();\n",
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

        GetInputScaleStr << llvm::formatv("    {0}_{1}_scale_vct = "
                             "{2}_desc.GetInputScale({3}_input_{4}_name);\n",
                                thatOpKey, inputSlot, thatOpKey, thatOpKey, inputSlot)
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
                   "    {0}_desc_main.SetInputScale(matched.at(\"{1}\")->arg()->name, "
                   "{2}_{3}_scale_vct);\n",
                                          opKey, actualName, thatOpKey, inputSlot)
                   .str();
      }

      os << "  if ( " << thatOpKey << "_is_quantized_op) {\n";
      os << GetInputScaleStr.str();
      os << "  }\n";

      SetInputScaleStrALL << "  if ( " << thatOpKey << "_is_quantized_op) {\n";
      SetInputScaleStrALL << SetInputScaleStr.str();
      SetInputScaleStrALL << "  }\n";
      SetInputScaleStr.clear();
    }

  } //结束处理SetInputScale的部分。

  //  os << "  op_desc.mutable_inputs()->clear();\n";
  //  os << "  op_desc.mutable_outputs()->clear();\n";

  auto targetPattern = dag;
  auto targetPatOp = targetPattern->getOp();
  auto &targetPatArgSlotNames = targetPatOp->getArgNames();
  auto &targetPatArgNames = targetPattern->getArgNames();
  auto &targetPatArgs = targetPattern->getArguments();

  os << llvm::formatv("  {0}_desc_main.SetType(\"{1}\");\n", opKey, targetPatOp->getType());

  unsigned actualArgSize = targetPatArgNames.size();
  std::stringstream IR_NODE_LINK_TO_code;

  for (unsigned i = 0; i < targetPatArgs.size(); i++) {
    //检查形参和实参类型是否匹配
    if (!targetPattern->isArgTypeCorrect(i)) {
      std::string msg = pat.getName() + ":";
      msg += targetPatOp->getType() + ":";
      msg += "arguments type mis match.";
      PrintFatalError(msg);
    }

    std::string argName = targetPatArgNames[i];
    bool refResult = pat.getTargetOpByOutputKey(argName) != nullptr;
    if (targetPatArgs[i]->getNodeType() == TDPatternNode::Var && !refResult) {
      os << llvm::formatv(
          "  {0}_desc_main.SetInput(\"{1}\", {matched.at(\"{2}\")->arg()->name});\n",
                          opKey, targetPatArgSlotNames[i], argName);

      IR_NODE_LINK_TO_code << llvm::formatv(
                                  "  IR_NODE_LINK_TO(matched.at(\"{0}\"), {1}_op_node);\n",
                                  argName, opKey).str();
    }
    else if (targetPatArgs[i]->getNodeType() == TDPatternNode::Var &&
               refResult) {
      std::string newNodeName = opKey + "_" + targetPatArgSlotNames[i];
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
          "  {0}_desc_main.SetInput(\"{1}\", {{\"{2}\"});\n",
                          opKey, targetPatArgSlotNames[i], newNodeName);

      IR_NODE_LINK_TO_code << llvm::formatv(
                                  "  IR_NODE_LINK_TO({0}_node, {1}_op_node);\n",
                                  newNodeName, opKey).str();
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
      //看这个嵌套的op是否指定的output key。
      std::string innerOpKey;
      if (innerOpNode->getDesignatedOutputKey() != "") {
        innerOpKey = innerOpNode->getDesignatedOutputKey();
      }
      else {
        innerOpKey = op2key[innerOpNode];
      }

      if (innerOpNode->getOp()->getType() == "DirectCompute") {

        //创建var node和新Tensor实例代码
        //        std::string fusion_bias_name = filter_name + "_conv_fusion_bias";
        //        auto* fusion_bias_node = graph->NewArgumentNode(fusion_bias_name);
        //        fusion_bias_node->arg()->is_weight = true;
        //        fusion_bias_node->arg()->type = LiteType::GetTensorTy(
        //            TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
        //
        //        auto* fusion_bias_t = scope->NewTensor(fusion_bias_name);
        //        fusion_bias_t->set_precision(paddle::lite_api::PrecisionType::kFloat);

        std::string newNodeName = opKey + "_" + targetPatArgSlotNames[i];
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
        // innerOpKey + "_val"是嵌套op的计算结果变量符号
        os << llvm::formatv("  {0}_t->CopyDataFrom({1});\n",
                            newNodeName, innerOpKey + "_val");
        os << llvm::formatv(
            "  {0}_desc_main.SetInput(\"{1}\", {{\"{2}\"});\n",
                            opKey, targetPatArgSlotNames[i], newNodeName);

        IR_NODE_LINK_TO_code << llvm::formatv(
                                    "  IR_NODE_LINK_TO({0}_node, {1}_op_node);\n",
                                    newNodeName, opKey).str();
      }
    }
  }

  for (unsigned i = 0, c = targetPatOp->getResNames().size(); i < c; i++) {
    os << llvm::formatv(
        "  {0}_desc_main.SetOutput(\"{1}\", {matched.at(\"{2}\")->arg()->name});\n",
                        opKey,
        targetPatOp->getResNames()[i], srcPatOutputNames[i]);
  }


  // handle AttrToSet
  auto &attrsToSet = pat.getAttrsToSet();
  if (attrsToSet.count(targetPatOp->getKey())) {
    for (auto &attr : attrsToSet.at(targetPatOp->getKey())) {
      genSetAttrCode(opKey + "_desc_main", attr, os);
    }
  }

  //handle attrToMap and attrToSet(by attr as argument)
  //TODO: attribute也可能是计算而来的。还未支持此种情况。
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
            "  if (matched.at(\"{0}\")->stmt()->op_info()->HasAttr(\"{1}\")) {{\n",
                       thatOpKey, thatAttrName);
        os << llvm::formatv(
            "    {0}_desc_main.SetAttr(\"{1}\", "
            "matched.at(\"{2}\")->stmt()->op_info()->GetAttr<{3}>(\"{4}\"));\n",
            opKey, thisAttrName, thatOpKey, dataType, thatAttrName);

        os << "  }\n";
        continue;
      }
      //否则，如果有值，采用参数值
      else if (static_cast<TDPatternAttrNode*>(targetPatArgs[i].get())
                   ->getAttr() != nullptr) {
        attr.value = static_cast<TDPatternAttrNode*>(targetPatArgs[i].get())
                         ->getAttr()->getValue();
      }
      //否则，使用默认值
      else {
          //attr.value = targetPattern->getSetOrDefaultAttrAtIndex(i)->getValue();
          continue;
      }
    }
    //实际参数个数少于形式参数个数，少的那部分必须用默认值
    else {
      //attr.value = targetPatOp->getArgumentAsAttrAtIndex(i)->getValue();
      continue;
    }
    genSetAttrCode(opKey + "_desc_main", attr, os);
  }

  // handle AttrToCopy
  auto &attrsToCp = pat.getAttrsToCopy();
  for (auto &attr : attrsToCp) {
    if (attr.attrName.at(0) == '#')
      continue;
    std::string fromOpKeyed = opDesignatedKey2opKey.at(attr.from);
    std::string toOpKeyed = opDesignatedKey2opKey.at(attr.to);

    os << llvm::formatv(
        "  if (matched.at(\"{0}\")->stmt()->op_info()->HasAttr(\"{1}\")) {\n",
        fromOpKeyed, attr.attrName);
    os << llvm::formatv(
        "    {0}_desc_main.SetAttr(\"{1}\", "
        "matched.at(\"{2}\")->stmt()->op_info()->GetAttr<{3}>(\"{4}\"));\n",
        opKey, attr.attrName, fromOpKeyed, attr.dataType, attr.attrName);
    os << "  }\n";
  }

  if (needCopyInputScale) {
    os << SetInputScaleStrALL.str();
  }

  //如果融合算子节点保留自源结点，这里不需要新建一个算子了，只需取回原来的算子
  if (retainKey != "") {
    std::string &allocatedKey = opDesignatedKey2opKey[retainKey];
    os << llvm::formatv("  matched.at(\"{0}\")->stmt()->ResetOp({1}_desc_main, valid_places);\n",
        allocatedKey, opKey);
    os << llvm::formatv("  auto* {0}_op_node = matched.at(\"{1}\");\n", opKey, allocatedKey);
  }
  else {
    os << llvm::formatv(
        "  auto {0}_op = LiteOpRegistry::Global().Create(\"{1}\");\n", opKey, opType);
    os << llvm::formatv("  {0}_op->Attach({1}_desc_main, scope);\n", opKey,
                        opKey);

    os << llvm::formatv("  auto* {0}_op_node = "
                        "graph->GraphCreateInstructNode({1}_op, valid_places);\n",
        opKey, opKey);
  }


  os << IR_NODE_LINK_TO_code.str();
  //TODO: 这里只需要连接到第一个输出节点么?
  if (isRoot) {
    os << llvm::formatv("  IR_NODE_LINK_TO({0}_op_node, matched.at(\"{1}\"));\n",
                        opKey, srcPatOutputNames[0]);
  }
  else {
  }

  return opKey;
}


void PdLiteGraphOptPassEmitter::handleTargetPatTopological(TDPattern &pat) {
  const auto &topo = pat.getTargetTopological();
  for (unsigned i = 0; i < topo.size(); i++) {
    TDPatternNode *node = topo[i];
    if (node->getNodeType() == TDPatternNode::Op) {
      auto opNode = static_cast<TDPatternOpNode*>(node);

      if (opNode->getOp()->getType() == "DirectCompute") {
        os << "// handle " << opNode->getOp()->getDirectComputeType() << "\n";
        handleDirectCompute(opNode, pat);
      }
      else {
        if (i != topo.size() - 1)
          llvm::PrintFatalError("multiple none direct compute op in "
                                "target pattern has not been supported now");
        else {
          os << "// handle " << opNode->getOp()->getType() << "\n";
          dfsResPatDag(opNode, true, pat);
        }
      }
    }
    //不需要处理var node或者attr node
    else {
    }
  }
}

std::pair<std::string, std::string>
PdLiteGraphOptPassEmitter::handleDirectCompute(TDPatternOpNode *dag,
                                                 TDPattern &pat) {

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
        computeVarSymbols.push_back(varName);
        computeVarSymbolIsPointer.push_back(true);
        //变量符号池里已经有这个符号了，说明这个值已经计算/获取过了，不需要再获取一次。
        if (computeVarSymbolPool.count(varName)) {
          continue;
        }
        computeVarSymbolPool.insert(varName);
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
      }
      //如果是绑定到另一个directCompute的结果，则
      else if (pat.getTargetOpByOutputKey(dag->getArgNames()[i])) {
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

void PdLiteGraphOptPassEmitter::EmitFuserHeader(TDPattern &pat) {

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

void PdLiteGraphOptPassEmitter::EmitFuserImpl(TDPattern &pat) {
  EmitBuildPatternMethod(pat);
  os << "\n";
  EmitBuildNewGraphMethod(pat);
}

void PdLiteGraphOptPassEmitter::EmitPassHeader(TDPattern &pat) {

  os << llvm::formatv("class {0}FusePass : public ProgramPass {{\n",
                      pat.getNameWithoutPat());
  os << R"(public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};
)";

}

void PdLiteGraphOptPassEmitter::EmitPassImpl(TDPattern &pat) {

  os << llvm::formatv("void {0}FusePass::Apply(\n",
                      pat.getNameWithoutPat());
  os << "    const std::unique_ptr<SSAGraph>& graph) {\n";
  os << llvm::formatv("  fusion::{0}Fuser fuser;\n",
                      pat.getNameWithoutPat());
  os << "  fuser(graph.get());\n";
  os << "}\n";
}

void PdLiteGraphOptPassEmitter::EmitRegisterPass(TDPattern &pat) {

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


void PdLiteGraphOptPassEmitter::EmitSingleFileHeader() {
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
    EmitSingleFileHeader();
  }
  OS << nameSpaceFusionBegin;
  OS << "\n";
  OS << directElementWiseCompute;
  OS << "\n";
  OS << rankChecker;
  OS << "\n";
  OS << nameSpaceFusionEnd;

  for (PdGraphOpt::TDPattern &pat : converter.getPatterns()) {
    OS << "//==============================================================================\n";
    OS << "//" << pat.getName() << "\n";
    OS << "//==============================================================================\n";
    OS << nameSpaceFusionBegin;
    OS << "\n";
    EmitFuserHeader(pat);
    OS << "\n";
    EmitFuserImpl(pat);
    OS << "\n";
    OS << nameSpaceFusionEnd;

    OS << "\n";

    OS << nameSpaceMirBegin;
    OS << "\n";
    EmitPassHeader(pat);
    OS << "\n";
    EmitPassImpl(pat);
    OS << "\n";
    OS << nameSpaceMirEnd;
    OS << "\n";
    EmitRegisterPass(pat);

    resetState();
  }
}

namespace llvm {

void EmitPaddleLiteGraphOptPass(RecordKeeper &RK, raw_ostream &OS) {
  PdLiteGraphOptPassEmitter(RK, OS).run(OS);
}

} // namespace llvm
