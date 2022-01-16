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
