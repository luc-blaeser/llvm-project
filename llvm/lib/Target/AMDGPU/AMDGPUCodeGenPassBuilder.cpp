//===- lib/Target/AMDGPU/AMDGPUCodeGenPassBuilder.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUCodeGenPassBuilder.h"
#include "AMDGPU.h"
#include "AMDGPUCtorDtorLowering.h"
#include "AMDGPUISelDAGToDAG.h"
#include "AMDGPUPerfHintAnalysis.h"
#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnifyDivergentExitNodes.h"
#include "SIFixSGPRCopies.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/ExpandVariadics.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/FlattenCFG.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/Scalar/LoopDataPrefetch.h"
#include "llvm/Transforms/Scalar/NaryReassociate.h"
#include "llvm/Transforms/Scalar/SeparateConstOffsetFromGEP.h"
#include "llvm/Transforms/Scalar/Sink.h"
#include "llvm/Transforms/Scalar/StraightLineStrengthReduce.h"
#include "llvm/Transforms/Scalar/StructurizeCFG.h"
#include "llvm/Transforms/Utils/FixIrreducible.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LowerSwitch.h"
#include "llvm/Transforms/Utils/UnifyLoopExits.h"
#include "llvm/Transforms/Vectorize/LoadStoreVectorizer.h"

using namespace llvm;
using namespace llvm::AMDGPU;

AMDGPUCodeGenPassBuilder::AMDGPUCodeGenPassBuilder(
    GCNTargetMachine &TM, const CGPassBuilderOption &Opts,
    PassInstrumentationCallbacks *PIC)
    : CodeGenPassBuilder(TM, Opts, PIC) {
  Opt.RequiresCodeGenSCCOrder = true;
  // Exceptions and StackMaps are not supported, so these passes will never do
  // anything.
  // Garbage collection is not supported.
  disablePass<StackMapLivenessPass, FuncletLayoutPass,
              ShadowStackGCLoweringPass>();
}

void AMDGPUCodeGenPassBuilder::addIRPasses(AddIRPass &addPass) const {
  // TODO: Missing AMDGPURemoveIncompatibleFunctions

  addPass(AMDGPUPrintfRuntimeBindingPass());
  if (LowerCtorDtor)
    addPass(AMDGPUCtorDtorLoweringPass());

  if (isPassEnabled(EnableImageIntrinsicOptimizer))
    addPass(AMDGPUImageIntrinsicOptimizerPass(TM));

  // This can be disabled by passing ::Disable here or on the command line
  // with --expand-variadics-override=disable.
  addPass(ExpandVariadicsPass(ExpandVariadicsMode::Lowering));

  addPass(AMDGPUAlwaysInlinePass());
  addPass(AlwaysInlinerPass());

  // TODO: Missing OpenCLEnqueuedBlockLowering

  // Runs before PromoteAlloca so the latter can account for function uses
  if (EnableLowerModuleLDS)
    addPass(AMDGPULowerModuleLDSPass(TM));

  if (TM.getOptLevel() > CodeGenOptLevel::None)
    addPass(InferAddressSpacesPass());

  // Run atomic optimizer before Atomic Expand
  if (TM.getOptLevel() >= CodeGenOptLevel::Less &&
      (AMDGPUAtomicOptimizerStrategy != ScanOptions::None))
    addPass(AMDGPUAtomicOptimizerPass(TM, AMDGPUAtomicOptimizerStrategy));

  addPass(AtomicExpandPass());

  if (TM.getOptLevel() > CodeGenOptLevel::None) {
    addPass(AMDGPUPromoteAllocaPass(TM));
    if (isPassEnabled(EnableScalarIRPasses))
      addStraightLineScalarOptimizationPasses(addPass);

    // TODO: Handle EnableAMDGPUAliasAnalysis

    // TODO: May want to move later or split into an early and late one.
    addPass(AMDGPUCodeGenPreparePass(TM));

    // TODO: LICM
  }

  Base::addIRPasses(addPass);

  // EarlyCSE is not always strong enough to clean up what LSR produces. For
  // example, GVN can combine
  //
  //   %0 = add %a, %b
  //   %1 = add %b, %a
  //
  // and
  //
  //   %0 = shl nsw %a, 2
  //   %1 = shl %a, 2
  //
  // but EarlyCSE can do neither of them.
  if (isPassEnabled(EnableScalarIRPasses))
    addEarlyCSEOrGVNPass(addPass);
}

void AMDGPUCodeGenPassBuilder::addCodeGenPrepare(AddIRPass &addPass) const {
  // AMDGPUAnnotateKernelFeaturesPass is missing here, but it will hopefully be
  // deleted soon.

  if (EnableLowerKernelArguments)
    addPass(AMDGPULowerKernelArgumentsPass(TM));

  // This lowering has been placed after codegenprepare to take advantage of
  // address mode matching (which is why it isn't put with the LDS lowerings).
  // It could be placed anywhere before uniformity annotations (an analysis
  // that it changes by splitting up fat pointers into their components)
  // but has been put before switch lowering and CFG flattening so that those
  // passes can run on the more optimized control flow this pass creates in
  // many cases.
  //
  // FIXME: This should ideally be put after the LoadStoreVectorizer.
  // However, due to some annoying facts about ResourceUsageAnalysis,
  // (especially as exercised in the resource-usage-dead-function test),
  // we need all the function passes codegenprepare all the way through
  // said resource usage analysis to run on the call graph produced
  // before codegenprepare runs (because codegenprepare will knock some
  // nodes out of the graph, which leads to function-level passes not
  // being run on them, which causes crashes in the resource usage analysis).
  addPass(AMDGPULowerBufferFatPointersPass(TM));

  Base::addCodeGenPrepare(addPass);

  if (isPassEnabled(EnableLoadStoreVectorizer))
    addPass(LoadStoreVectorizerPass());

  // LowerSwitch pass may introduce unreachable blocks that can cause unexpected
  // behavior for subsequent passes. Placing it here seems better that these
  // blocks would get cleaned up by UnreachableBlockElim inserted next in the
  // pass flow.
  addPass(LowerSwitchPass());
}

void AMDGPUCodeGenPassBuilder::addPreISel(AddIRPass &addPass) const {
  const bool LateCFGStructurize = AMDGPUTargetMachine::EnableLateStructurizeCFG;
  const bool DisableStructurizer = AMDGPUTargetMachine::DisableStructurizer;
  const bool EnableStructurizerWorkarounds =
      AMDGPUTargetMachine::EnableStructurizerWorkarounds;

  if (TM.getOptLevel() > CodeGenOptLevel::None)
    addPass(FlattenCFGPass());

  if (TM.getOptLevel() > CodeGenOptLevel::None)
    addPass(SinkingPass());

  addPass(AMDGPULateCodeGenPreparePass(TM));

  // Merge divergent exit nodes. StructurizeCFG won't recognize the multi-exit
  // regions formed by them.

  addPass(AMDGPUUnifyDivergentExitNodesPass());

  if (!LateCFGStructurize && !DisableStructurizer) {
    if (EnableStructurizerWorkarounds) {
      addPass(FixIrreduciblePass());
      addPass(UnifyLoopExitsPass());
    }

    addPass(StructurizeCFGPass(/*SkipUniformRegions=*/false));
  }

  addPass(AMDGPUAnnotateUniformValuesPass());

  if (!LateCFGStructurize && !DisableStructurizer) {
    addPass(SIAnnotateControlFlowPass(TM));

    // TODO: Move this right after structurizeCFG to avoid extra divergence
    // analysis. This depends on stopping SIAnnotateControlFlow from making
    // control flow modifications.
    addPass(AMDGPURewriteUndefForPHIPass());
  }

  addPass(LCSSAPass());

  if (TM.getOptLevel() > CodeGenOptLevel::Less)
    addPass(AMDGPUPerfHintAnalysisPass(TM));

  // FIXME: Why isn't this queried as required from AMDGPUISelDAGToDAG, and why
  // isn't this in addInstSelector?
  addPass(RequireAnalysisPass<UniformityInfoAnalysis, Function>());
}

void AMDGPUCodeGenPassBuilder::addAsmPrinter(AddMachinePass &addPass,
                                             CreateMCStreamer) const {
  // TODO: Add AsmPrinter.
}

Error AMDGPUCodeGenPassBuilder::addInstSelector(AddMachinePass &addPass) const {
  addPass(AMDGPUISelDAGToDAGPass(TM));
  addPass(SIFixSGPRCopiesPass());
  addPass(SILowerI1CopiesPass());
  return Error::success();
}

void AMDGPUCodeGenPassBuilder::addEarlyCSEOrGVNPass(AddIRPass &addPass) const {
  if (TM.getOptLevel() == CodeGenOptLevel::Aggressive)
    addPass(GVNPass());
  else
    addPass(EarlyCSEPass());
}

void AMDGPUCodeGenPassBuilder::addStraightLineScalarOptimizationPasses(
    AddIRPass &addPass) const {
  if (isPassEnabled(EnableLoopPrefetch, CodeGenOptLevel::Aggressive))
    addPass(LoopDataPrefetchPass());

  addPass(SeparateConstOffsetFromGEPPass());

  // ReassociateGEPs exposes more opportunities for SLSR. See
  // the example in reassociate-geps-and-slsr.ll.
  addPass(StraightLineStrengthReducePass());

  // SeparateConstOffsetFromGEP and SLSR creates common expressions which GVN or
  // EarlyCSE can reuse.
  addEarlyCSEOrGVNPass(addPass);

  // Run NaryReassociate after EarlyCSE/GVN to be more effective.
  addPass(NaryReassociatePass());

  // NaryReassociate on GEPs creates redundant common expressions, so run
  // EarlyCSE after it.
  addPass(EarlyCSEPass());
}

bool AMDGPUCodeGenPassBuilder::isPassEnabled(const cl::opt<bool> &Opt,
                                             CodeGenOptLevel Level) const {
  if (Opt.getNumOccurrences())
    return Opt;
  if (TM.getOptLevel() < Level)
    return false;
  return Opt;
}
