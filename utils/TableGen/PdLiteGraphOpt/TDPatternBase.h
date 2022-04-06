//
// Created by 骆荟州 on 2022/4/6.
//

#ifndef LLVM_TDPATTERNBASE_H
#define LLVM_TDPATTERNBASE_H

#include "llvm/TableGen/Error.h"
#include "TDOpArgument.h"
#include <map>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <queue>

namespace PdGraphOpt {

/**
 * 属性操作（断言、设置、映射）等的基础结构
 */
struct AttrOperationBase {
  std::string attrName;
  std::string dataType;

  void setDataType(std::string dType) {
    //全部用小写字母表示类型。
    std::transform(dType.begin(), dType.end(), dType.begin(), ::tolower);
    this->dataType = std::move(dType);
  }
};

/**
 * 记录需要给目标结点设置的属性条目
 */
struct AttrToSet : AttrOperationBase {
  std::string value;
  std::string target;

  std::string getDescription() {
    std::stringstream ss;
    ss << "attrName:" << attrName << "(" << dataType << ") "
       << "value:" << value << "set for:" << target;
    return ss.str();
  }

  bool checkIntegrity() {
    return attrName != "" && dataType != "" && value != "" && target != "";
  }
};

/**
 * 记录需要从源结点拷贝到目标结点的属性条目
 */
struct AttrToCopy : AttrOperationBase {

  std::string from;
  std::string to;

  std::string getDescription() {
    std::stringstream ss;
    ss << "attrName:" << attrName << "(" << dataType << ") "
       << "from:" << from << "  to:" << to;
    return ss.str();
  }

  bool checkIntegrity() {
    return attrName != "" && dataType != "" && from != "" && to != "";
  }
};

/**
 * 记录需要施加断言的结点和施加的断言
 */
struct AttrToAssert : AttrOperationBase {

  std::string target;
  std::string value;
  std::string customAssert;
  bool useCustomAssert{false};

  std::string getDescription() {
    std::stringstream ss;
    ss << "attrName:" << attrName << "(" << dataType << ") "
       << "value:" << value << "  target:" << target;
    return ss.str();
  }

  bool checkIntegrity() {
    return attrName != "" && dataType != "" && target != "" && value != "";
  }

};

/**
 * 记录需要对结点应用的自定义teller
 */
struct CustomTeller {
  std::string target;
  std::string name;
  std::string teller;
};

struct ExtraCondition {

  std::string conditionType;
  std::string sourceNode;
  std::string targetNode;
  std::string dataType;
  std::string value1;
  std::string value2;
  std::string value3;
};


/**
 * 内存中以树的形式存储td中描述的pattern，该类为树的结点类。
 */
class TDPatternNode {
public:
  enum NodeType { Op, Var, Attr };

  virtual ~TDPatternNode() {}
  virtual NodeType getNodeType() = 0;
};

/**
 * 内存中以树的形式存储td中描述的pattern，该类为树的变量结点类。
 */
class TDPatternVarNode : public TDPatternNode {
public:
  TDPatternVarNode(std::shared_ptr<TDVariable> var) : var(var) {}

  NodeType getNodeType() override { return NodeType::Var; }

  TDVariable* getVar() const { return var.get(); }

private:
  std::shared_ptr<TDVariable> var;
};

/**
 * 内存中以树的形式存储td中描述的pattern，该类为树的属性结点类。
 */
class TDPatternAttrNode: public TDPatternNode {
public:

  TDPatternAttrNode(std::shared_ptr<TDAttribute> attr) : attr(attr) {}

  NodeType getNodeType() override { return NodeType::Attr; }

  TDAttribute* getAttr() const { return attr.get(); }

private:
  std::shared_ptr<TDAttribute> attr;
};

/**
 * 内存中以树的形式存储td中描述的pattern，该类为树的op结点类。
 */
class TDPatternOpNode : public TDPatternNode {
public:
  TDPatternOpNode(std::shared_ptr<TDOperator> &op,
                  std::string& opKey,
                  std::vector<std::unique_ptr<TDPatternNode>> arguments,
                  std::vector<std::string> argNames)
      : op(op), designatedOutputKey(opKey),
        arguments(std::move(arguments)), argNames(std::move(argNames)) {
  }

  NodeType getNodeType() override { return NodeType::Op; }

  TDOperator* getOp() { return op.get(); }

  std::vector<std::unique_ptr<TDPatternNode>> &getArguments() {
    return arguments;
  }

  std::vector<std::string> &getArgNames() { return argNames; }

  std::pair<TDPatternNode *, std::string> getArgAndName(unsigned index) {
    if (index > argNames.size()) {
      llvm::PrintFatalError("Index out of bound while getting arguments "
                            "from `TDPatternOpNode`.");
    }
    return std::make_pair(arguments[index].get(), argNames[index]);
  }

  TDAttribute* getSetOrDefaultAttrAtIndex(unsigned index) {


    if (index >= op->getArgNames().size()) return nullptr;
    if (index >= argNames.size()) {
      return op->getArgumentAsAttrAtIndex(index).get();
    }
    else {
      auto attr =
          static_cast<TDPatternAttrNode*>(arguments[index].get())->getAttr();
      if (attr != nullptr) return attr;
      return op->getArgumentAsAttrAtIndex(index).get();
    }
  }

  std::string getArgSlotNameByActualArgName(std::string actualArgName) {
    auto pos = std::find(argNames.begin(),
                         argNames.end(), actualArgName);
    if (pos == argNames.end()) {
      return "";
    }
    long index = pos - argNames.begin();
    return op->getArgNames()[index];
  }

  long getIndexByActualArgName(std::string name) {
    auto pos = std::find(argNames.begin(),
                         argNames.end(), name);

    if (pos == argNames.end()) {
      return -1;
    }
    return pos - argNames.begin();
  }

  bool isArgTypeCorrect(unsigned index) {
    unsigned slotNum = op->getArgNames().size();
    unsigned realArgNum = argNames.size();
    if (index < 0) return true;
    if (index >= slotNum) return false;
    if (index >= realArgNum && index < slotNum) return true;

    if (op->getArgumentTypeAtIndex(index) == TDOpArgument::variable) {
      if (arguments[index]->getNodeType() == Var) return true;
      if (arguments[index]->getNodeType() == Op) return true;
      return false;
    }
    if (op->getArgumentTypeAtIndex(index) == TDOpArgument::attribute) {
      if (arguments[index]->getNodeType() == Attr) return true;
      return false;
    }
    return false;
  }

  std::string getDesignatedOutputKey() {
    return designatedOutputKey;
  }

private:
  std::shared_ptr<TDOperator> op;
  std::string designatedOutputKey{""};
  std::vector<std::unique_ptr<TDPatternNode>> arguments;
  std::vector<std::string> argNames;
};

}

#endif // LLVM_TDPATTERNBASE_H
