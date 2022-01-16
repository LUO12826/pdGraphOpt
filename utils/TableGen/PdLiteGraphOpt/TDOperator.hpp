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
#include "TDVariable.hpp"


namespace PdGraphOpt {

class TDOperator {
  std::string key;
  std::string type;
  std::string summary;
  std::string description;

  std::vector<std::shared_ptr<TDVariable>> arguments;
  std::vector<std::string> argNames;
  std::vector<std::shared_ptr<TDVariable>> results;
  std::vector<std::string> resNames;

public:
  explicit TDOperator(std::string &key, std::string &type): key(key), type(type) {

  }
  void setKey(std::string key) {
    this->key = key;
  }

  std::string getKey() {
    return key;
  }

  std::string getType() {
    return type;
  }

  std::vector<std::string>& getArgNames() {
      return argNames;
  };

  std::vector<std::string>& getResNames() {
    return resNames;
  };

  void setArguments(std::vector<std::shared_ptr<TDVariable>> &args,
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
    std::string rep = "type:" + type + "\n";
    rep += "key:" + key + "\n";
    rep += "args: ";
    for(unsigned i = 0, count = arguments.size(); i < count; i++) {
      rep += argNames[i] + ":" + arguments[i]->getType() + ", ";
    }
    rep += "\n";
    rep += "res: ";
    for(unsigned i = 0, count = results.size(); i < count; i++) {
      rep += resNames[i] + ":" + results[i]->getType() + ", ";
    }
    rep += "\n";
    return rep;
  }

};

}



#endif /* TDOperator_h */
