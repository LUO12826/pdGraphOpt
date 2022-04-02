//
// Created by 骆荟州 on 2022/4/1.
//

#ifndef LLVM_TDOPARGUMENT_H
#define LLVM_TDOPARGUMENT_H

namespace PdGraphOpt {

class TDOpArgument {
public:

  enum ArgumentType {
    variable,
    attribute
  };

  virtual ~TDOpArgument() {}
  virtual ArgumentType getArgumentType() = 0;
};

class TDVariable: public TDOpArgument {
  std::string name;
  std::string type;

  bool isWeight{false};
  bool isPersistable{false};

public:

  TDVariable(std::string &name, std::string &type) {
    this->name = name;
    this->type = type;
  }

  ArgumentType getArgumentType() override {
    return TDOpArgument::variable;
  }

  std::string getType() {
    return type;
  }

  std::string getName() {
    return name;
  }

  void setIsWeight(bool yesOrNo) {
    this->isWeight = yesOrNo;
  }

  void setIsPersist(bool yesOrNo) {
    this->isPersistable = yesOrNo;
  }

  bool getIsWeight() {
    return isWeight;
  }

  bool getIsPersist() {
    return isPersistable;
  }
};

class TDAttribute: public TDOpArgument {
  std::string name;
  std::string type;
  std::string value;

public:
  ArgumentType getArgumentType() override {
    return TDOpArgument::attribute;
  }

  TDAttribute(std::string &type, std::string value) {
    this->type = type;
    this->value = value;
  }

  TDAttribute(std::string &name, std::string &type, std::string value) {
    this->name = name;
    this->type = type;
    this->value = value;
  }

  //attribute的数据类型。返回的是小写。
  std::string getType() {
    std::string t = type;
    std::transform(t.begin(), t.end(), t.begin(), tolower);
    return t;
  }

  std::string getName() {
    return name;
  }

  std::string getValue() {
    return value;
  }

  int getValueAsInt() {
    return std::stoi(value);
  }

  std::string getValueAsQuotedString() {
    return "\"" + value + "\"";
  }
};
}
#endif // LLVM_TDOPARGUMENT_H