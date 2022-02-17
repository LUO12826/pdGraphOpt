//
//  TDPattern.h
//  llvm-tblgen
//
//  Created by 骆荟州 on 2022/1/8.
//

#ifndef TDPattern_h
#define TDPattern_h

#include <memory>
#include <map>

namespace PdGraphOpt {

struct AttrOperationRecordBase {
  std::string attrName;
  std::string dataType;

  void setDataType(std::string dType) {
    //全部用小写字母表示类型。
    std::transform(dType.begin(), dType.end(), dType.begin(), ::tolower);
    this->dataType = std::move(dType);
  }
};

struct AttrToSet: AttrOperationRecordBase {
  std::string value;
  std::string target;

  std::string stringDesc() {
    std::stringstream ss;
    ss << "attrName:" << attrName << "(" << dataType << ") "
       << "value:" << value << "set for:" << target;
    return ss.str();
  }
};

struct AttrToCopy: AttrOperationRecordBase {

  std::string fromKeyedOp;
  std::string toKeyedOp;

  std::string stringDesc() {
    std::stringstream ss;
    ss << "attrName:" << attrName << "(" << dataType << ") "
        << "from:" << fromKeyedOp << "  to:" << toKeyedOp;
    return ss.str();
  }
};

struct AttrToAssert: AttrOperationRecordBase {

  std::string target;
  std::string value;
  std::string customAssert;
  bool useCustomAssert{false};

  std::string stringDesc() {
    std::stringstream ss;
    ss << "attrName:" << attrName << "(" << dataType << ") "
       << "value:" << value << "  target:" << target;
    return ss.str();
  }
};

/**
 * 内存中以树的形式存储td中描述的pattern，该类为树的结点类。
 */
class TDPatternNode {
public:
  enum NodeType {
    Op, Var
  };

  virtual ~TDPatternNode() {}
  virtual NodeType getNodeType() = 0;

};

class TDPatternOpNode: public TDPatternNode {
public:
  TDPatternOpNode(std::shared_ptr<TDOperator> &op,
                  std::vector<std::unique_ptr<TDPatternNode>> arguments,
                  std::vector<std::string> argNames)
      :op(op), arguments(std::move(arguments)), argNames(std::move(argNames)) {}

  NodeType getNodeType() override {
    return Op;
  }

  std::shared_ptr<TDOperator> getOp() {
    return op;
  }

  std::vector<std::unique_ptr<TDPatternNode>>& getArguments() {
    return arguments;
  }

  std::vector<std::string>& getArgNames() {
    return argNames;
  }

private:
  std::shared_ptr<TDOperator> op;
  std::vector<std::unique_ptr<TDPatternNode>> arguments;
  std::vector<std::string> argNames;
};


class TDPatternVarNode: public TDPatternNode {
public:
  TDPatternVarNode(std::shared_ptr<TDVariable> &var): var(var) {}

  NodeType getNodeType() override {
    return Var;
  }

  std::shared_ptr<TDVariable> getVar() {
    return var;
  }

private:
  std::shared_ptr<TDVariable> var;
};

/**
 * 用于表示td文件中描述的一个变换pattern
 */
class TDPattern {
  //pattern的名字
  std::string name;
  //指向源pattern。源pattern和目标pattern都以树的结构表示
  std::unique_ptr<TDPatternOpNode> sourcePatternRoot;
  //指向目标pattern。
  std::unique_ptr<TDPatternOpNode> targetPatternRoot;

  std::vector<AttrToCopy> attrsToCopy;
  std::vector<AttrToSet> attrsToSet;
  std::map<std::string, std::vector<AttrToAssert>> attrsToAssert;

  std::vector<std::string> conditionAttribute;

  bool hasMultipleTargets{false};
  int benefitDelta{0};

public:
  TDPattern(std::string name, std::unique_ptr<TDPatternOpNode> &&sourcePatternRoot,
            std::unique_ptr<TDPatternOpNode> &&targetPatternRoot):
            name(name),
            sourcePatternRoot(std::move(sourcePatternRoot)),
            targetPatternRoot(std::move(targetPatternRoot)) {}

  void setAttrsToCopy(std::vector<AttrToCopy> &&attrs) {
    this->attrsToCopy = std::move(attrs);
  }

  std::vector<AttrToCopy>& getAttrsToCopy() {
    return this->attrsToCopy;
  }

  std::vector<std::string>& getConditionAttribute() {
    return this->conditionAttribute;
  }

  void setAttrsToSet(std::vector<AttrToSet> &&attrs) {
    this->attrsToSet = std::move(attrs);
  }

  std::vector<AttrToSet>& getAttrsToSet() {
    return this->attrsToSet;
  }

  void setAttrsToAssert(std::map<std::string, std::vector<AttrToAssert>> &&attrs) {
    this->attrsToAssert = std::move(attrs);
  }

  std::map<std::string, std::vector<AttrToAssert>>& getAttrsToAssert() {
    return this->attrsToAssert;
  }

  std::string getName() {
    return name;
  }

  std::string getNameWithoutPat() {
    std::string n = name;
    return n.erase(name.length() - 3);
  }

  TDPatternOpNode* getSourcePatternRoot() {
      return sourcePatternRoot.get();
  }

  TDPatternOpNode* getTargetPatternRoot() {
    return targetPatternRoot.get();
  }

  std::string getStringRepresentation() {
    std::string rep = "pattern name: " + name + "\n";
    return rep;
  }
};

}


#endif /* TDPattern_h */
