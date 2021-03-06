//
// Created by 骆荟州 on 2022/1/16.
//

#ifndef LLVM_RECORDCONVERTER_H
#define LLVM_RECORDCONVERTER_H

#include "TDOperator.hpp"
#include "TDOpArgument.h"
#include "TDPattern.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <typeinfo>
#include <algorithm>

namespace PdGraphOpt {

class RecordConverter {
  llvm::RecordKeeper &Records;

  std::vector<TDPattern> patterns;

  // 存放生成的TDOperator实例，key是record的名字
  std::map<std::string, std::shared_ptr<TDOperator>> OpCache;

  // 存放生成的TDVariable实例，key是record的名字
  std::map<std::string, std::shared_ptr<TDVariable>> VarCache;

  // 存放生成的TDAttribute实例，key是record的名字
  std::map<std::string, std::shared_ptr<TDAttribute>> AttrCache;

  TDPatternOpNode *buildPatternNodeFromDagInit(llvm::DagInit *dagInit) {
    auto opInDag = llvm::dyn_cast<llvm::DefInit>(dagInit->getOperator());
    if (!opInDag) {
      llvm::PrintFatalError("Pattern dag operator must be a Record");
    }
    auto opRecord = opInDag->getDef();
    auto op = OpCache.at(opRecord->getName().str());
    std::string opKey = dagInit->getNameStr().str();
    std::vector<std::string> argNames;
    std::vector<std::unique_ptr<TDPatternNode>> argNodes;
    for (unsigned i = 0, count = dagInit->getNumArgs(); i < count; i++) {
      llvm::Init *argInit = dagInit->getArg(i);

      argNames.push_back(dagInit->getArgNameStr(i).str());
      if (llvm::isa<llvm::DagInit>(argInit)) {
        auto argNodeInit = llvm::dyn_cast<llvm::DagInit>(argInit);
        TDPatternOpNode *newOpNode = buildPatternNodeFromDagInit(argNodeInit);
        argNodes.push_back(std::unique_ptr<TDPatternOpNode>(newOpNode));
      }
      // 如果dag此处的值为空，说明不关心var或attr的类型限制，只关心通过其名字建立的绑定关系
      else if (llvm::isa<llvm::UnsetInit>(argInit)) {
        if (op->getArgumentTypeAtIndex(i) == TDOpArgument::attribute) {
          argNodes.push_back(std::make_unique<TDPatternAttrNode>(nullptr));
        }
        else {
          argNodes.push_back(std::make_unique<TDPatternVarNode>(nullptr));
        }
      }
      else {
        auto argNodeRec = llvm::dyn_cast<llvm::DefInit>(argInit)->getDef();
        if (argNodeRec->isSubClassOf("Var")) {
          auto var = VarCache.at(argNodeRec->getName().str());
          argNodes.push_back(std::make_unique<TDPatternVarNode>(var));
        }
        else if (argNodeRec->isSubClassOf("OpAttr")) {
          auto attr = AttrCache.at(argNodeRec->getName().str());
          argNodes.push_back(std::make_unique<TDPatternAttrNode>(attr));
        }
      }
    }

    return new TDPatternOpNode(op, opKey, std::move(argNodes),
                               std::move(argNames));
  }

  std::vector<std::string> getStringList(llvm::Record *rec,
                                         llvm::StringRef field) {
    std::vector<std::string> list;
    const auto& originalList
        = rec->getValueAsListOfStrings(field);
    for (auto &s : originalList) {
      list.push_back(s.str());
    }
    return list;
  }

public:
  std::vector<TDPattern> getPatterns() { return std::move(patterns); }

  explicit RecordConverter(llvm::RecordKeeper &records) : Records(records) {

    //先把所有Var类型的Record读一遍
    auto varRecs = Records.getAllDerivedDefinitions("Var");
    for (llvm::Record *rec : varRecs) {
      std::string type = rec->getValueAsString("type").str();
      std::string name = rec->getValueAsString("name").str();
      auto var = std::make_shared<TDVariable>(name, type);
      var->setIsPersist(rec->getValueAsBit("isPersistable"));
      var->setIsWeight(rec->getValueAsBit("isWeight"));
      var->setDataType(rec->getValueAsString("dataType").str());
      VarCache.insert(std::make_pair(rec->getName().str(), var));
    }

    //先把所有Attr类型的Record读一遍
    auto attrRecs = Records.getAllDerivedDefinitions("OpAttr");
    for (llvm::Record *rec : attrRecs) {
      std::string type = rec->getValueAsDef("dataType")->getName().str();
      std::string name = rec->getValueAsString("name").str();
      std::string value = rec->getValueAsString("value").str();
      auto attr = std::make_shared<TDAttribute>(name, type, value);
      AttrCache.insert(std::make_pair(rec->getName().str(), attr));
    }

    //然后把所有Op类型的Record读一遍
    auto opRecs = Records.getAllDerivedDefinitions("Op");
    for (llvm::Record *rec : opRecs) {
      std::string type = rec->getValueAsString("type").str();
      std::string key = rec->getValueAsString("key").str();
      auto op = std::make_shared<TDOperator>(key, type);
      op->setTypeIsVariable(rec->getValueAsBit("typeIsVariable"));

      std::vector<std::string> names;
      std::vector<std::shared_ptr<TDOpArgument>> args;

      auto opArgs = rec->getValueAsDag("arguments");
      auto opRes = rec->getValueAsDag("results");
      for (unsigned i = 0, count = opArgs->getNumArgs(); i < count; i++) {
        auto opArgRec =
            llvm::dyn_cast<llvm::DefInit>(opArgs->getArg(i))
            ->getDef();
        auto recName = opArgRec->getName().str();

        names.push_back(opArgs->getArgNameStr(i).str());
        if (opArgRec->isSubClassOf("OpAttr")) {
          args.push_back(AttrCache.at(recName));
        } else if (opArgRec->isSubClassOf("Var")) {
          args.push_back(VarCache.at(recName));
        }

      }

      op->setArguments(args, names);
      names.clear();
      std::vector<std::shared_ptr<TDVariable>> res;
      for (unsigned i = 0, count = opRes->getNumArgs(); i < count; i++) {
        auto defInit = llvm::dyn_cast<llvm::DefInit>(opRes->getArg(i));
        auto varRecName = defInit->getDef()->getName().str();
        names.push_back(opRes->getArgNameStr(i).str());
        res.push_back(VarCache.at(varRecName));
      }

      op->setResult(res, names);
      // 特殊情况
      if (op->getType() == "DirectCompute") {
        op->setDirectComputeType(rec->getValueAsString("directComputeType").str());
      }
      OpCache.insert(std::make_pair(rec->getName().str(), op));
    }

    //最后构建Pattern
    auto patRecs = Records.getAllDerivedDefinitions("Pat");
    for (llvm::Record *rec : patRecs) {
      auto sourcePat =
          buildPatternNodeFromDagInit(rec->getValueAsDag("sourcePattern"));
      //这里目前只处理了只有一种target pattern的情况
      auto targetPatDag = llvm::dyn_cast<llvm::DagInit>(
          rec->getValueAsListInit("targetPatterns")->getElement(0));
      auto targetPat = buildPatternNodeFromDagInit(targetPatDag);

      patterns.emplace_back(rec->getName().str(),
                            std::unique_ptr<TDPatternOpNode>(sourcePat),
                            std::unique_ptr<TDPatternOpNode>(targetPat));
      TDPattern &newPat = patterns.back();

      newPat.kernelName = rec->getValueAsString("kernelName").str();
      // read `attrToCopy`
      auto *attrsToCpList = rec->getValueAsListInit("attrToCopy");
      for (unsigned i = 0, c = attrsToCpList->size(); i < c; i++) {
        auto *dagInitItem =
            llvm::dyn_cast<llvm::DagInit>(attrsToCpList->getElement(i));
        if (!dagInitItem) {
          llvm::PrintFatalError(
              "Every item in attrsToCopy or attrsToSet list must be a dag.");
        }
        AttrToCopy attr;
        attr.attrName = dagInitItem->getArg(0)->getAsUnquotedString();
        attr.setDataType(dagInitItem->getArg(1)->getAsUnquotedString());
        attr.from = dagInitItem->getArg(2)->getAsUnquotedString();
        attr.to = dagInitItem->getArg(3)->getAsUnquotedString();


        if(!attr.checkIntegrity()) {

        }

        if(attr.attrName.at(0) == '#') {
          if(attr.attrName == "#INPUT_SCALE") {
            newPat.needCopyInputScale = true;
          }
        }
        newPat.attrsToCopy.push_back(std::move(attr));
      }

      // read `attrToSet`
      auto *attrsToSetList = rec->getValueAsListInit("attrToSet");
      for (unsigned i = 0, c = attrsToSetList->size(); i < c; i++) {
        auto *dagInitItem =
            llvm::dyn_cast<llvm::DagInit>(attrsToSetList->getElement(i));
        //assert(dagInitItem != nullptr && "Every item in attrsToCopy or attrsToSet list must be a dag.");
        if (!dagInitItem) {
          llvm::PrintFatalError(
              "Every item in attrsToCopy or attrsToSet list must be a dag.");
        }
        AttrToSet attr;
        attr.target = dagInitItem->getArg(0)->getAsUnquotedString();
        attr.attrName = dagInitItem->getArg(1)->getAsUnquotedString();
        attr.setDataType(dagInitItem->getArg(2)->getAsUnquotedString());
        attr.value = dagInitItem->getArg(3)->getAsUnquotedString();
        newPat.attrsToSet[attr.target].push_back(std::move(attr));
      }

      // read `extraAssertions`
      auto extraAssertionsList = rec->getValueAsListOfDefs("extraAssertions");
      for (auto assertRec : extraAssertionsList) {

        if (!assertRec->isSubClassOf("ExtraCondition")) {
          llvm::PrintFatalError(
              "Every item in extraAssertions list must be a ExtraCondition instance.");
        }
        ExtraCondition cond;
        cond.conditionType = assertRec->getValueAsString("conditionType").str();
        cond.sourceNode = assertRec->getValueAsString("sourceNode").str();
        cond.targetNode = assertRec->getValueAsString("targetNode").str();
        cond.dataType = assertRec->getValueAsString("dataType").str();
        cond.value1 = assertRec->getValueAsString("value1").str();
        cond.value2 = assertRec->getValueAsString("value2").str();
        cond.value3 = assertRec->getValueAsString("value3").str();
        newPat.extraAssertions[cond.conditionType].push_back(std::move(cond));
      }

      //read `attrToAssert`
      auto *attrsToAssertList = rec->getValueAsListInit("attrToAssert");
      for (unsigned i = 0, c = attrsToAssertList->size(); i < c; i++) {
        auto *dagInitItem =
            llvm::dyn_cast<llvm::DagInit>(attrsToAssertList->getElement(i));

        if (!dagInitItem) {
          llvm::PrintFatalError(
              "Every item in attrToAssert list must be a dag.");
        }
        AttrToAssert attr;
        std::string assertorType = dagInitItem->getOperator()->getAsUnquotedString();

        attr.target = dagInitItem->getArg(0)->getAsUnquotedString();
        attr.attrName = dagInitItem->getArg(1)->getAsUnquotedString();
        attr.setDataType(dagInitItem->getArg(2)->getAsUnquotedString());
        if(assertorType == "assertor") {
          attr.value = dagInitItem->getArg(3)->getAsUnquotedString();
        } else if (assertorType == "customAssertor") {
          attr.useCustomAssert = true;
          attr.customAssert = dagInitItem->getArg(3)->getAsUnquotedString();
        }

        newPat.attrsToAssert[attr.target].push_back(std::move(attr));
      }

      //read `CustomTeller`
      auto *tellersList = rec->getValueAsListInit("customTeller");
      for (unsigned i = 0, c = tellersList->size(); i < c; i++) {
        auto *dagInitItem =
            llvm::dyn_cast<llvm::DagInit>(tellersList->getElement(i));

        if (!dagInitItem) {
          llvm::PrintFatalError(
              "Every item in attrToAssert list must be a dag.");
        }
        CustomTeller teller;

        teller.target = dagInitItem->getArg(0)->getAsUnquotedString();
        teller.teller = (dagInitItem->getArg(1)->getAsUnquotedString());

        newPat.customTellers[teller.target].push_back(std::move(teller));
      }

      //read `conditionFlags`
      newPat.conditionFlags = getStringList(rec, "conditionFlags");
      //read `bindTargets`
      newPat.bindTargets = getStringList(rec, "bindTargets");
      //read `excludeTargets`
      newPat.excludeTargets = getStringList(rec, "excludeTargets");

    } //end build Pattern
  }
};
} // namespace PdGraphOpt

#endif // LLVM_RECORDCONVERTER_H
