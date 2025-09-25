from shir import types, layout, config, bit_utils
import torch
from typing import Tuple, Optional, Dict
from functools import reduce
from itertools import chain

_supported_ops = {}

def register_lowering(key):
  def _magic(lowering):
    assert key not in _supported_ops, f"Operation {key} is repeatedly registered"
    _supported_ops[key] = lowering
    return lowering   # allows stacking this decorator
  return _magic

def fetch_lowering(key):
  return _supported_ops.get(key)

shin = torch.ops.shir_intrinsic
aten = torch.ops.aten
prims = torch.ops.prims

@register_lowering(shin.rnn.default)
class OperatorRNN:
  @staticmethod
  def supports(x, ih, hh, b, tanh_or_relu) -> bool:
    return False

  @staticmethod
  def lower(x, ih, hh, b, tanh_or_relu) -> str:
    assert False, "TODO"

@register_lowering(shin.lstm.default)
class OperatorLSTM:
  @staticmethod
  def supports(x, ihs, hhs, bs, proj) -> bool:
    wii, wif, wig, wio = ihs
    whi, whf, whg, who = hhs
    bi, bf, bg, bo = bs

    # TODO: if extra validation is necessary...
    return True

  @staticmethod
  def lower(x, ihs, hhs, bs, proj) -> str:
    wii, wif, wig, wio = ihs
    whi, whf, whg, who = hhs
    bi, bf, bg, bo = bs

    # this is a 3D tensor: [batch x sequence length x input size]
    tests, length, _ = x.meta.get("val").shape
    print("HEREEE!!!", x.meta.get("val").shape)
    if proj is not None:
      # projection is used
      pass

    hiddenSize, inputSize = wii.meta.get("val").shape
    print("hidden size:", hiddenSize, " input size:", inputSize)
    bitWidth = 16
    fraction = 10
    precision = 2**10
    
    return f"LSTM({bitWidth}, {inputSize}, {hiddenSize}, {length}, {tests}, {x},\
      {wii}, {wif}, {wig}, {wio}, {whi}, {whf}, {whg}, {who}, {bi}, {bf}, {bg}, {bo})"

  
  @staticmethod
  def scanLSTM(x, wii, wif, wig, wio, whi, whf, whg, who, bi, bf, bg, bo, hiddenSize, inputSize, bitWidth, fraction, precision) -> str:
    intType = f"SigmoidInt({bitWidth})" 
    i = "ParamUse(inputParam)"
    h = f"DropVector(ParamUse(stateParam),0,{hiddenSize})"
    c = f"VectorToOrderedStream(MapVector(Registered.asFunction(), DropVector(ParamUse(stateParam), {hiddenSize}, 0)))"
    bfReg = f"MapOrderedStream(Registered.asFunction(), {bf})"
    biReg = f"MapOrderedStream(Registered.asFunction(), {bi})"
    boReg = f"MapOrderedStream(Registered.asFunction(), {bo})"
    bcReg = f"MapOrderedStream(Registered.asFunction(), {bg})"
    init = OperatorLSTM.initState(hiddenSize*2, intType)

    wf = OperatorLSTM.storeInBram(wif, bitWidth)
    wi = OperatorLSTM.storeInBram(wii, bitWidth)
    wo = OperatorLSTM.storeInBram(wio, bitWidth)
    wc = OperatorLSTM.storeInBram(wig, bitWidth)

    uf = OperatorLSTM.storeInBram(whf, bitWidth)
    ui = OperatorLSTM.storeInBram(whi, bitWidth)
    uo = OperatorLSTM.storeInBram(who, bitWidth)
    uc = OperatorLSTM.storeInBram(whg, bitWidth)




    F = f"TypeChecker.check({OperatorLSTM.sigmoidStream(OperatorLSTM.gateOp(uf, h, wf, i, bitWidth, fraction, bfReg, intType), intType, precision)})"
    I = f"TypeChecker.check({OperatorLSTM.sigmoidStream(OperatorLSTM.gateOp(ui, h, wi, i, bitWidth, fraction, biReg, intType), intType, precision)})"
    Cprime = f"TypeChecker.check({OperatorLSTM.tanhStream(OperatorLSTM.gateOp(uc, h, wc, i, bitWidth, fraction, bcReg, intType), intType, precision)})"
    O = f"TypeChecker.check({OperatorLSTM.sigmoidStream(OperatorLSTM.gateOp(uo, h, wo, i, bitWidth, fraction, boReg, intType), intType, precision)})"
    Cnew = f'''
      OrderedStreamToVector(TypeChecker.check({OperatorLSTM.elementWiseAdd(
        OperatorLSTM.elementWiseMul(F, c, intType, bitWidth, fraction),
        OperatorLSTM.elementWiseMul(I, Cprime, intType, bitWidth, fraction),
        intType)}
      ))
    '''


    
    stateParamDef = f"val stateParam = TypeChecker.check(ParamDef(VectorType({intType}, ArithType({hiddenSize}*2))))"
    inputParamDef = f"val inputParam = TypeChecker.check(ParamDef(VectorType({intType}, ArithType({inputSize}))))"
    cParamDef = f"val CParam = ParamDef(TypeChecker.check({Cnew}).t)"
    
    
    functionDef = f'''
      {{
        val stateParam = TypeChecker.check(ParamDef(VectorType({intType}, ArithType({hiddenSize}*2))))
        val inputParam = TypeChecker.check(ParamDef(VectorType({intType}, ArithType({inputSize}))))
        val CParam = ParamDef(TypeChecker.check({Cnew}).t)
        ArchLambdas(
        Seq(stateParam, inputParam),
        Let(
          CParam,
          ConcatVector(Tuple(
            MapVector({{
                val p = ParamDef(TupleType({intType}, {intType}))
                ArchLambda(
                  p,
                  ClipBankersRound(MulInt(ParamUse(p)), {fraction}, {bitWidth}-{fraction})
                )
              }},
              ZipVector(Tuple2({O}, {OperatorLSTM.tanhVector(f"ParamUse(CParam)", intType, precision)}))
            ),
            ParamUse(CParam)
          ))
          ,
          TypeChecker.check({Cnew})
          )
        )
      }}
      '''
    
    scanLSTM = f'''
      TypeChecker.check(MapOrderedStream({{
        val in = ParamDef(TypeChecker.check({x}).t.asInstanceOf[OrderedStreamTypeT].et)
        ArchLambda(
          in,
          TypeChecker.check(ScanOrderedStream(
              {functionDef},
              {init},
              ParamUse(in)
            ))
          )}},
        TypeChecker.check({x})))
    '''

  @staticmethod
  def storeInBram(matrix, bitWidth) -> str:
    return f'''
      mem.BufferStreamOfVecInBlockRam(MapOrderedStream(
        OrderedStreamToVector.asFunction(),
        {OperatorLSTM.resizeMatrix(matrix, bitWidth)}))
    '''

  @staticmethod
  def resizeMatrix(matrix, bitWidth) -> str : 
    return f'''
      MapOrderedStream({{
        val p = ParamDef(TypeChecker.check({matrix}).t.asInstanceOf[OrderedStreamTypeT].et)
        ArchLambda(
          p,
          MapOrderedStream(
          ResizeInteger.asFunction(Seq(None), Seq({bitWidth})),
          TypeChecker.check(ParamUse(p))
        ))}},
        {matrix}
      )'''
  

  @staticmethod
  def gateOp(hiddenMatrix, h, inputMatrix, x, bitWidth, fraction, bias, intType) -> str:
    hMVM = OperatorLSTM.mvm(hiddenMatrix, h, bitWidth, fraction)
    inMVM = OperatorLSTM.mvm(inputMatrix, x, bitWidth, fraction)
    addMVMs = OperatorLSTM.elementWiseAdd(hMVM, inMVM, intType)
    gate = OperatorLSTM.elementWiseAdd(addMVMs, bias, intType)
    return f"TypeChecker.check({gate})"
  
  @staticmethod
  def initState(size, intType) -> str:
    return f'''
    MapVector(Registered.asFunction(), ConstantVector(Seq.fill(size)(0), Some({intType})))
    '''
  
  @staticmethod
  def mvm(matrix, vector, bitWidth, fraction) -> str:
    
    notTruncated = f'''MapOrderedStream({{
      val param = ParamDef(TypeChecker.check({matrix}).t.asInstanceOf[OrderedStreamTypeT].et)
      ArchLambda(
        param,
        FoldVector.sum(
          MapVector({{
            val inParam = ParamDef(TupleType(SignedIntType({bitWidth}), SignedIntType({bitWidth})))
            ArchLambda(
              inParam,
              Registered(ClipBankersRound(MulInt(TypeChecker.check(Registered(ParamUse(inParam)))), {fraction}, {bitWidth} - {fraction}))
            )
          }},
            ZipVector(Tuple(ParamUse(param), {vector}))
      )))}},
      {matrix}
    )'''

    foldBitwidth = f"TypeChecker.check({notTruncated}).t.asInstanceOf[OrderedStreamTypeT].et.bitWidth.ae"
    return f'''
      MapOrderedStream({{
        val param = ParamDef(TypeChecker.check({notTruncated}).t.asInstanceOf[OrderedStreamTypeT].et)
        ArchLambda(
          param,
          Registered(ClipInt(ParamUse(param), {foldBitwidth} - {bitWidth}))
        )}},
        {notTruncated}
      )'''
  
  @staticmethod
  def tanhStream(stm, intType, precision) -> str:
    return f'''
      MapOrderedStream({{
        val p = ParamDef({intType})
        ArchLambda(
          p,
          TanhInt(ParamUse(p), {precision})
        )}},
        {stm})
    '''
  
  @staticmethod
  def tanhVector(vec, intType, precision) -> str:
    return f'''
      MapVector({{
        val p = ParamDef({intType})
        ArchLambda(
          p,
          TanhInt(ParamUse(p), {precision})
        )}},
        {vec}
      )
    '''
  
  @staticmethod
  def sigmoidStream(stm, intType, precision) -> str:
    return f'''
      MapOrderedStream({{
        val p = ParamDef({intType})
        ArchLambda(
          p,
          SigmoidInt(ParamUse(p), {precision})
        )}},
        {stm}
    )
    '''
  
  @staticmethod
  def elementWiseAdd(stm1, stm2, intType) -> str:
    return f'''
      MapOrderedStream({{
        val p = ParamDef(TupleType({intType},{intType}))
        ArchLambda(
          p,
          Registered(ClipInt(AddInt(Registered(ParamUse(p))), 1))
        )}},
        Zip2OrderedStream(Tuple({stm1}, {stm2}))
    )
    '''
  
  @staticmethod
  def elementWiseMul(stm1, stm2, intType, bitWidth, fraction) -> str:
    return f'''
      MapOrderedStream({{
        val p = ParamDef(TupleType({intType}, {intType}))
        ArchLambda(
          p,
          (ClipBankersRound(MulInt(ParamUse(p)), {fraction}, {bitWidth}-{fraction}))
        )}},
        Zip2OrderedStream(Tuple(TypeChecker.check({stm1}), TypeChecker.check({stm2})))
    )
    '''
  

  

  




@register_lowering(aten.view.default)
class OperatorView:
  @staticmethod
  def supports(a, shape) -> bool:
    return True

  @staticmethod
  def lower(a, shape) -> str:
    # reshape the metatensor to get the resulting shape.
    #
    # do this instead of using shape directly because we might have unresolved
    # (-1) lengths...
    fk = a.meta.get("val")
    nd = fk.ndim
    ys = fk.reshape(shape).shape

    # the shir expression is just a simple join-all + split-all
    se = str(a)
    for _ in range(1, nd):
      se = f"JoinOrderedStream({se})"
    for y in reversed(ys[1:]):
      se = f"SplitOrderedStream({se}, {y})"
    return se


