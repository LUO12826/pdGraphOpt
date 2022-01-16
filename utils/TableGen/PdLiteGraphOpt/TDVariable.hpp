//
//  TDVariable.h
//  llvm-tblgen
//
//  Created by 骆荟州 on 2022/1/8.
//

#ifndef TDVariable_h
#define TDVariable_h
namespace PdGraphOpt {

class TDVariable {
  std::string name;
  std::string type;

public:

  TDVariable(std::string &name, std::string &type) {
    this->name = name;
    this->type = type;
  }

  std::string getType() {
      return type;
  }

  std::string getName() {
    return name;
  }
};

}


#endif /* TDVariable_h */
