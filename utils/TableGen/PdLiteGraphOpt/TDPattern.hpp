//
//  TDPattern.h
//  llvm-tblgen
//
//  Created by 骆荟州 on 2022/1/8.
//

#ifndef TDPattern_h
#define TDPattern_h

#include <memory>

namespace PdGraphOpt {


struct AttrToSet {
  std::string attrName;
  std::string dataType;
  std::string value;

  std::string stringDesc() {
    std::stringstream ss;
    ss << "attrName:" << attrName << "(" << dataType << ") "
       << "value:" << value;
    return ss.str();
  }
};

struct AttrToCopy {
  std::string attrName;
  std::string dataType;
  std::string fromKeyedOp;
  std::string toKeyedOp;

  std::string stringDesc() {
    std::stringstream ss;
    ss << "attrName:" << attrName << "(" << dataType << ") "
        << "from:" << fromKeyedOp << "  to:" << toKeyedOp;
    return ss.str();
  }
};

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

class TDPattern {
  std::string name;
  std::unique_ptr<TDPatternOpNode> sourcePatternRoot;
  std::unique_ptr<TDPatternOpNode> targetPatternRoot;

  std::vector<AttrToCopy> attrsToCopy;
  std::vector<AttrToSet> attrsToSet;

  bool hasMultipleTargets{false};
  int benefitDelta{0};

public:
  TDPattern(std::string name, std::unique_ptr<TDPatternOpNode> &&sourcePatternRoot,
            std::unique_ptr<TDPatternOpNode> &&targetPatternRoot):
            name(name),
            sourcePatternRoot(std::move(sourcePatternRoot)),
            targetPatternRoot(std::move(targetPatternRoot)) {}

  std::string getStringRepresentation() {
    std::string rep = "pattern name: " + name + "\n";
    return rep;
  }

  void setAttrsToCopy(std::vector<AttrToCopy> &&attrs) {
    this->attrsToCopy = std::move(attrs);
  }

  std::vector<AttrToCopy>& getAttrsToCopy() {
    return this->attrsToCopy;
  }

  void setAttrsToSet(std::vector<AttrToSet> &&attrs) {
    this->attrsToSet = std::move(attrs);
  }

  std::vector<AttrToSet>& getAttrsToSet() {
    return this->attrsToSet;
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
};

}


#endif /* TDPattern_h */