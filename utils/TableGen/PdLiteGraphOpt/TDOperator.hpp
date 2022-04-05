//
//  TDOperator.h
//  llvm-tblgen
//
//  Created by 骆荟州 on 2022/1/8.
//

#ifndef TDOperator_h
#define TDOperator_h

#include <vector>
#include <memory>
#include <sstream>
#include "TDOpArgument.h"

namespace PdGraphOpt {

class TDOperator {
  std::string key;
  std::string type;
  std::string directComputeType{""};
  bool typeIsVariable{false};
  std::string summary{""};
  std::string description{""};

  std::vector<std::shared_ptr<TDOpArgument>> arguments;
  std::vector<std::string> argNames;
  std::vector<std::shared_ptr<TDVariable>> results;
  std::vector<std::string> resNames;

public:
  explicit TDOperator(std::string &key, std::string &type): key(key), type(type) {
  }

  void setDirectComputeType(std::string type) {
    this->directComputeType = type;
  }

  std::string getDirectComputeType() {
    return this->directComputeType;
  }

  void setKey(std::string key) {
    this->key = key;
  }

  std::string getKey() {
    return key;
  }

  void setTypeIsVariable(bool yesOrNo) {
    this->typeIsVariable = yesOrNo;
  }

  bool getTypeIsVariable() {
    return this->typeIsVariable;
  }

  std::string getTypeAuto() {
      return typeIsVariable ? type : "\"" + type + "\"";
  }

  std::string getType() {
    return type;
  }

  std::string getTypeWithQuote() {
    return "\"" + type + "\"";
  }

  std::vector<std::string>& getArgNames() {
      return argNames;
  };

  TDOpArgument::ArgumentType getArgumentTypeAtIndex(unsigned index) {
    return arguments[index]->getArgumentType();
  }

  const auto& getArguments(){
    return arguments;
  }

  std::shared_ptr<TDVariable> getArgumentAsVarAtIndex(unsigned index) {
    assert(
        arguments[index]->getArgumentType() == TDOpArgument::variable
           && "try to get an argument from TDOperator as variable but get attribute");
    return std::static_pointer_cast<TDVariable>(arguments[index]);
  }

  std::shared_ptr<TDAttribute> getArgumentAsAttrAtIndex(unsigned index) {
    assert(
        arguments[index]->getArgumentType() == TDOpArgument::attribute
           && "try to get an argument from TDOperator as attribute but get variable");
    return std::static_pointer_cast<TDAttribute>(arguments[index]);
  }

  int getIndexInArgNames(std::string arg) {
      auto iter = std::find(argNames.begin(), argNames.end(), arg);
      if(iter == argNames.end()) return -1;
      return iter - argNames.begin();
  }

  std::vector<std::string>& getResNames() {
    return resNames;
  };

  void setArguments(std::vector<std::shared_ptr<TDOpArgument>> &args,
                    std::vector<std::string> &argNames) {
    this->arguments = args;
    this->argNames = argNames;
  }

  void setResult(std::vector<std::shared_ptr<TDVariable>> &res,
                 std::vector<std::string> &resNames) {
    this->results = res;
    this->resNames = resNames;
  }

  std::string getStringRepresentation() {
    std::stringstream rep;
    std::stringstream attrs;
    rep << "type:" << type << "\n";
    rep << "key:" << key << "\n";
    rep << "args: ";
    for(unsigned i = 0, count = arguments.size(); i < count; i++) {
      if (arguments[i]->getArgumentType() == TDOpArgument::variable) {
        rep << argNames[i] << ":"
            << std::static_pointer_cast<TDVariable>(arguments[i])->getType()
            << ", ";
      }
      else {
        attrs << argNames[i] << ":"
            << std::static_pointer_cast<TDAttribute>(arguments[i])->getDataType()
            << ", ";
      }

    }
    rep << "\n";
    rep << "attrs: " << attrs.str() << "\n";
    rep << "res: ";
    for(unsigned i = 0, count = results.size(); i < count; i++) {
      rep << resNames[i] << ":" << results[i]->getType() << ", ";
    }
    rep << "\n";
    return rep.str();
  }

};
}



#endif /* TDOperator_h */
