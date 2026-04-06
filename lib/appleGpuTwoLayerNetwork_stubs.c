#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include <dispatch/dispatch.h>

#import <Metal/Metal.h>

static id<MTLDevice> device;
static id<MTLCommandQueue> commandQueue;
static id<MTLComputePipelineState> predictBatchPipelineState;
static id<MTLComputePipelineState> trainEpochsPipelineState;

#include "appleGpuTwoLayerNetworkMetallib.inc"

static id<MTLComputePipelineState> makePipelineState(NSString *functionName) {
  NSError *error = nil;
  dispatch_data_t metallibData = dispatch_data_create(
      appleGpuTwoLayerNetworkMetallibData,
      appleGpuTwoLayerNetworkMetallibData_len,
      dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
      DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  id<MTLLibrary> library =
      [device newLibraryWithData:metallibData error:&error];

  if (library == nil)
    caml_failwith([[error localizedDescription] UTF8String]);

  id<MTLFunction> function = [library newFunctionWithName:functionName];

  if (function == nil)
    caml_failwith("Missing Metal function");

  id<MTLComputePipelineState> pipelineState =
      [device newComputePipelineStateWithFunction:function error:&error];

  if (pipelineState == nil)
    caml_failwith([[error localizedDescription] UTF8String]);

  return pipelineState;
}

static void ensureContext(void) {
  if (device != nil)
    return;

  @autoreleasepool {
    device = MTLCreateSystemDefaultDevice();

    if (device == nil)
      caml_failwith("Apple GPU backend requires a Metal device");

    commandQueue = [device newCommandQueue];

    if (commandQueue == nil)
      caml_failwith("Apple GPU backend failed to create command queue");

    predictBatchPipelineState = makePipelineState(@"predictBatchKernel");
    trainEpochsPipelineState = makePipelineState(@"trainEpochsKernel");
  }
}

static id<MTLBuffer> copyFloatArrayToBuffer(value floatArrayValue) {
  mlsize_t count = Wosize_val(floatArrayValue);
  id<MTLBuffer> buffer =
      [device newBufferWithLength:count * sizeof(float)
                          options:MTLResourceStorageModeShared];
  float *bufferValues = (float *)[buffer contents];

  for (mlsize_t index = 0; index < count; ++index)
    bufferValues[index] = (float)Double_field(floatArrayValue, index);

  return buffer;
}

static void copyBufferToFloatArray(id<MTLBuffer> buffer,
                                   value floatArrayValue) {
  mlsize_t count = Wosize_val(floatArrayValue);
  const float *bufferValues = (const float *)[buffer contents];

  for (mlsize_t index = 0; index < count; ++index)
    Store_double_field(floatArrayValue, index, (double)bufferValues[index]);
}

static value copyBufferToNewFloatArray(id<MTLBuffer> buffer, NSUInteger count) {
  CAMLparam0();
  CAMLlocal1(floatArrayValue);

  const float *bufferValues = (const float *)[buffer contents];

  floatArrayValue = caml_alloc_float_array((mlsize_t)count);

  for (NSUInteger index = 0; index < count; ++index)
    Store_double_field(floatArrayValue, index, (double)bufferValues[index]);

  CAMLreturn(floatArrayValue);
}

static void dispatchThreads(id<MTLComputeCommandEncoder> encoder,
                            id<MTLComputePipelineState> pipelineState,
                            NSUInteger count) {
  NSUInteger width = [pipelineState threadExecutionWidth];

  if (width == 0)
    width = 1;

  MTLSize threadgroupSize = MTLSizeMake(MIN(width, MAX(count, 1u)), 1, 1);
  MTLSize gridSize = MTLSizeMake(MAX(count, 1u), 1, 1);

  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

static void commitAndWait(id<MTLCommandBuffer> commandBuffer) {
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  if ([commandBuffer error] != nil)
    caml_failwith([[[commandBuffer error] localizedDescription] UTF8String]);
}

CAMLprim value caml_apple_gpu_predict_batch(
    value hiddenWeightsValue, value hiddenBiasesValue, value outputWeightsValue,
    value outputBiasesValue, value inputsBatchValue, value exampleCountValue) {
  CAMLparam5(hiddenWeightsValue, hiddenBiasesValue, outputWeightsValue,
             outputBiasesValue, inputsBatchValue);
  CAMLxparam1(exampleCountValue);
  ensureContext();

  @autoreleasepool {
    NSUInteger exampleCount = (NSUInteger)Int_val(exampleCountValue);
    id<MTLBuffer> hiddenWeightsBuffer =
        copyFloatArrayToBuffer(hiddenWeightsValue);
    id<MTLBuffer> hiddenBiasesBuffer =
        copyFloatArrayToBuffer(hiddenBiasesValue);
    id<MTLBuffer> outputWeightsBuffer =
        copyFloatArrayToBuffer(outputWeightsValue);
    id<MTLBuffer> outputBiasesBuffer =
        copyFloatArrayToBuffer(outputBiasesValue);
    id<MTLBuffer> inputsBatchBuffer = copyFloatArrayToBuffer(inputsBatchValue);
    id<MTLBuffer> outputsBuffer =
        [device newBufferWithLength:MAX(exampleCount, 1u) * sizeof(float)
                            options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    uint32_t exampleCountBytes = (uint32_t)exampleCount;

    [encoder setComputePipelineState:predictBatchPipelineState];
    [encoder setBuffer:hiddenWeightsBuffer offset:0 atIndex:0];
    [encoder setBuffer:hiddenBiasesBuffer offset:0 atIndex:1];
    [encoder setBuffer:outputWeightsBuffer offset:0 atIndex:2];
    [encoder setBuffer:outputBiasesBuffer offset:0 atIndex:3];
    [encoder setBuffer:inputsBatchBuffer offset:0 atIndex:4];
    [encoder setBuffer:outputsBuffer offset:0 atIndex:5];
    [encoder setBytes:&exampleCountBytes length:sizeof(uint32_t) atIndex:6];

    dispatchThreads(encoder, predictBatchPipelineState, exampleCount);

    [encoder endEncoding];

    commitAndWait(commandBuffer);
    CAMLreturn(copyBufferToNewFloatArray(outputsBuffer, exampleCount));
  }
}

CAMLprim value caml_apple_gpu_predict_batch_bytecode(value *values, int count) {
  return caml_apple_gpu_predict_batch(values[0], values[1], values[2],
                                      values[3], values[4], values[5]);
}

CAMLprim value caml_apple_gpu_train_epochs(
    value epochCountValue, value learningRateValue, value hiddenWeightsValue,
    value hiddenBiasesValue, value outputWeightsValue, value outputBiasesValue,
    value trainingInputsValue, value expectedOutputsValue) {
  CAMLparam5(epochCountValue, learningRateValue, hiddenWeightsValue,
             hiddenBiasesValue, outputWeightsValue);
  CAMLxparam3(outputBiasesValue, trainingInputsValue, expectedOutputsValue);
  ensureContext();

  @autoreleasepool {
    id<MTLBuffer> hiddenWeightsBuffer =
        copyFloatArrayToBuffer(hiddenWeightsValue);
    id<MTLBuffer> hiddenBiasesBuffer =
        copyFloatArrayToBuffer(hiddenBiasesValue);
    id<MTLBuffer> outputWeightsBuffer =
        copyFloatArrayToBuffer(outputWeightsValue);
    id<MTLBuffer> outputBiasesBuffer =
        copyFloatArrayToBuffer(outputBiasesValue);
    id<MTLBuffer> trainingInputsBuffer =
        copyFloatArrayToBuffer(trainingInputsValue);
    id<MTLBuffer> expectedOutputsBuffer =
        copyFloatArrayToBuffer(expectedOutputsValue);
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    uint32_t epochCount = (uint32_t)Int_val(epochCountValue);
    uint32_t exampleCount = (uint32_t)Wosize_val(expectedOutputsValue);
    float learningRate = (float)Double_val(learningRateValue);

    [encoder setComputePipelineState:trainEpochsPipelineState];
    [encoder setBuffer:hiddenWeightsBuffer offset:0 atIndex:0];
    [encoder setBuffer:hiddenBiasesBuffer offset:0 atIndex:1];
    [encoder setBuffer:outputWeightsBuffer offset:0 atIndex:2];
    [encoder setBuffer:outputBiasesBuffer offset:0 atIndex:3];
    [encoder setBuffer:trainingInputsBuffer offset:0 atIndex:4];
    [encoder setBuffer:expectedOutputsBuffer offset:0 atIndex:5];
    [encoder setBytes:&epochCount length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&exampleCount length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&learningRate length:sizeof(float) atIndex:8];

    dispatchThreads(encoder, trainEpochsPipelineState, 1);

    [encoder endEncoding];

    commitAndWait(commandBuffer);
    copyBufferToFloatArray(hiddenWeightsBuffer, hiddenWeightsValue);
    copyBufferToFloatArray(hiddenBiasesBuffer, hiddenBiasesValue);
    copyBufferToFloatArray(outputWeightsBuffer, outputWeightsValue);
    copyBufferToFloatArray(outputBiasesBuffer, outputBiasesValue);
    CAMLreturn(Val_unit);
  }
}

CAMLprim value caml_apple_gpu_train_epochs_bytecode(value *values, int count) {
  return caml_apple_gpu_train_epochs(values[0], values[1], values[2], values[3],
                                     values[4], values[5], values[6],
                                     values[7]);
}
