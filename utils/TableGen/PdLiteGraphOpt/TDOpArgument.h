//
// Created by 骆荟州 on 2022/4/1.
//

#ifndef LLVM_TDOPARGUMENT_H
#define LLVM_TDOPARGUMENT_H

namespace PdGraphOpt {

/**
 * 表示tablegen中op的参数，包括变量和属性
 */
class TDOpArgument {
public:

  enum ArgumentType {
    variable,
    attribute
  };

  virtual ~TDOpArgument() {}
  virtual ArgumentType getArgumentType() = 0;
};

/**
 * 表示tablegen中op的变量参数
 */
class TDVariable: public TDOpArgument {
  std::string name;
  std::string type;
  std::string dataType;
  std::string value;

  bool isWeight{false};
  bool isPersistable{false};

public:

  TDVariable(std::string &name, std::string &type) {
    this->name = name;
    this->type = type;
  }

  void setDataType(std::string typ) {
      this->dataType = typ;
  }

  std::string getDataType() {
      return dataType;
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

/**
 * 表示tablegen中op的属性参数
 */
class TDAttribute: public TDOpArgument {
  std::string name;
  std::string dataType;
  std::string value;

public:
  ArgumentType getArgumentType() override {
    return TDOpArgument::attribute;
  }

  TDAttribute(std::string &type, std::string &value) {
    this->dataType = type;
    this->value = value;
  }

  TDAttribute(std::string &name, std::string &type, std::string &value) {
    this->name = name;
    this->dataType = type;
    this->value = value;
  }

  //attribute的数据类型。返回的是小写。
  std::string getDataType() {
    std::string t = dataType;
    std::transform(t.begin(), t.end(), t.begin(), tolower);
    return t;
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
