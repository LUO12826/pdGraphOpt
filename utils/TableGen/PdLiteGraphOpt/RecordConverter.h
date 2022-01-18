//
// Created by 骆荟州 on 2022/1/16.
//

#ifndef LLVM_RECORDCONVERTER_H
#define LLVM_RECORDCONVERTER_H

#include "TDOperator.hpp"
#include "TDPattern.hpp"
#include "TDVariable.hpp"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/Error.h"
#include "llvm/Support/Debug.h"
#include <typeinfo>

namespace PdGraphOpt {

class RecordConverter {
  llvm::RecordKeeper &Records;

  std::vector<TDPattern> patterns;

  // 存放生成的TDOperator实例，key是record的名字
  std::map<std::string, std::shared_ptr<TDOperator>> OpCache;

  // 存放生成的TDVariable实例，key是record的名字
  std::map<std::string, std::shared_ptr<TDVariable>> VarCache;

  TDPatternOpNode* buildPatternNodeFromDagInit(llvm::DagInit *dagInit) {
    auto opInDag = llvm::dyn_cast<llvm::DefInit>(dagInit->getOperator());
    if(!opInDag) {
      llvm::PrintFatalError("Pattern dag operator must be a Record");
    }
    auto opRecord = opInDag->getDef();
    auto op = OpCache.at(opRecord->getName().str());

    std::vector<std::string> argNames;
    std::vector<std::unique_ptr<TDPatternNode>> argNodes;
    for(unsigned i = 0, count = dagInit->getNumArgs(); i < count; i++) {
      llvm::Init *argInit = dagInit->getArg(i);

      argNames.emplace_back(dagInit->getArgNameStr(i).str());
      if(llvm::isa<llvm::DagInit>(argInit)) {
        auto argNodeInit = llvm::dyn_cast<llvm::DagInit>(argInit);
        TDPatternOpNode *newOpNode = buildPatternNodeFromDagInit(argNodeInit);
        argNodes.emplace_back(std::unique_ptr<TDPatternOpNode>(newOpNode));
      }
      else {
        auto argNodeRec = llvm::dyn_cast<llvm::DefInit>(argInit)->getDef();
        auto var = VarCache.at(argNodeRec->getName().str());

        argNodes.emplace_back(std::make_unique<TDPatternVarNode>(var));
      }
    }

    return new TDPatternOpNode(op, std::move(argNodes), std::move(argNames));
  }

public:

  std::vector<TDPattern> getPatterns() {
    return std::move(patterns);
  }

  explicit RecordConverter(llvm::RecordKeeper &records)
      : Records(records) {

    //先把所有Var类型的Record读一遍
    auto varRecs = Records.getAllDerivedDefinitions("Var");
    for(llvm::Record *rec : varRecs) {
      std::string type = rec->getValueAsString("type").str();
      std::string name = rec->getValueAsString("name").str();
      auto var = std::make_shared<TDVariable>(name, type);
      var->setIsPersist(rec->getValueAsBit("isPersist"));
      var->setIsWeight(rec->getValueAsBit("isWeight"));
      VarCache.insert(std::make_pair(rec->getName().str(), var));
    }

    //然后把所有Op类型的Record读一遍
    auto opRecs = Records.getAllDerivedDefinitions("Op");
    for(llvm::Record *rec : opRecs) {
      std::string type = rec->getValueAsString("type").str();
      std::string key = rec->getValueAsString("key").str();
      auto op = std::make_shared<TDOperator>(key, type);
      op->setTypeIsVariable(rec->getValueAsBit("typeIsVariable"));

      std::vector<std::string> names;
      std::vector<std::shared_ptr<TDVariable>> args;

      auto opArgs = rec->getValueAsDag("arguments");
      auto opRes = rec->getValueAsDag("results");
      for(unsigned i = 0, count = opArgs->getNumArgs(); i < count; i++) {
        auto defInit = llvm::dyn_cast<llvm::DefInit>(opArgs->getArg(i));
        auto varRecName = defInit->getDef()->getName().str();
        names.emplace_back(opArgs->getArgNameStr(i).str());
        args.push_back(VarCache.at(varRecName));
      }

      op->setArguments(args, names);
      names.clear();
      args.clear();

      for(unsigned i = 0, count = opRes->getNumArgs(); i < count; i++) {
        auto defInit = llvm::dyn_cast<llvm::DefInit>(opRes->getArg(i));
        auto varRecName = defInit->getDef()->getName().str();
        names.emplace_back(opRes->getArgNameStr(i).str());
        args.push_back(VarCache.at(varRecName));
      }

      op->setResult(args, names);
      OpCache.insert(std::make_pair(rec->getName().str(), op));
    }

    //最后构建Pattern
    auto patRecs = Records.getAllDerivedDefinitions("Pat");
    for(llvm::Record *rec : patRecs) {
      auto sourcePat = buildPatternNodeFromDagInit(
          rec->getValueAsDag("sourcePattern"));
      auto targetPatDag = llvm::dyn_cast<llvm::DagInit>(
            rec->getValueAsListInit("targetPatterns")->getElement(0)
          );
      auto targetPat = buildPatternNodeFromDagInit(targetPatDag);

      patterns.emplace_back(rec->getName().str(),
                            std::unique_ptr<TDPatternOpNode>(sourcePat),
                            std::unique_ptr<TDPatternOpNode>(targetPat)
          );

      std::vector<AttrToCopy> attrsToCopy;
      std::vector<AttrToSet> attrsToSet;

      //read `attrToCopy`
      auto *attrsToCpList = rec->getValueAsListInit("attrToCopy");
      for(unsigned i = 0, c = attrsToCpList->size(); i < c; i++) {
        auto *dagInitItem = llvm::dyn_cast<llvm::DagInit>(attrsToCpList->getElement(i));
        if(!dagInitItem) {
          llvm::PrintFatalError("Every item in attrsToCopy or attrsToSet list must be a dag.");
        }
        AttrToCopy attr;
        attr.attrName = dagInitItem->getArg(0)->getAsUnquotedString();
        attr.dataType = dagInitItem->getArg(1)->getAsUnquotedString();
        attr.fromKeyedOp = dagInitItem->getArg(2)->getAsUnquotedString();
        attr.toKeyedOp = dagInitItem->getArg(3)->getAsUnquotedString();
        attrsToCopy.push_back(attr);
      }

      //read `attrToSet`
      auto *attrsToSetList = rec->getValueAsListInit("attrToSet");
      for(unsigned i = 0, c = attrsToSetList->size(); i < c; i++) {
        auto *dagInitItem = llvm::dyn_cast<llvm::DagInit>(attrsToSetList->getElement(i));
        if(!dagInitItem) {
          llvm::PrintFatalError("Every item in attrsToCopy or attrsToSet list must be a dag.");
        }
        AttrToSet attr;
        attr.attrName = dagInitItem->getArg(0)->getAsUnquotedString();
        attr.dataType = dagInitItem->getArg(1)->getAsUnquotedString();
        attr.value = dagInitItem->getArg(2)->getAsUnquotedString();
        attrsToSet.push_back(attr);
      }

      patterns.back().setAttrsToCopy(std::move(attrsToCopy));
      patterns.back().setAttrsToSet(std::move(attrsToSet));
    }
  }
};
}

#endif // LLVM_RECORDCONVERTER_H