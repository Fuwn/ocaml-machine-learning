#include <metal_stdlib>

using namespace metal;

inline float sigmoid(float value) { return 1.0f / (1.0f + exp(-value)); }

kernel void predictBatchKernel(const device float *hiddenWeights [[buffer(0)]],
                               const device float *hiddenBiases [[buffer(1)]],
                               const device float *outputWeights [[buffer(2)]],
                               const device float *outputBiases [[buffer(3)]],
                               const device float *inputsBatch [[buffer(4)]],
                               device float *outputs [[buffer(5)]],
                               constant uint &exampleCount [[buffer(6)]],
                               uint exampleIndex [[thread_position_in_grid]]) {
  if (exampleIndex >= exampleCount)
    return;

  uint inputOffset = exampleIndex * 2;
  float input0 = inputsBatch[inputOffset];
  float input1 = inputsBatch[inputOffset + 1];

  float hidden0 = sigmoid(hiddenBiases[0] + (hiddenWeights[0] * input0) +
                          (hiddenWeights[1] * input1));
  float hidden1 = sigmoid(hiddenBiases[1] + (hiddenWeights[2] * input0) +
                          (hiddenWeights[3] * input1));

  outputs[exampleIndex] =
      sigmoid(outputBiases[0] + (outputWeights[0] * hidden0) +
              (outputWeights[1] * hidden1));
}

kernel void trainEpochsKernel(device float *hiddenWeights [[buffer(0)]],
                              device float *hiddenBiases [[buffer(1)]],
                              device float *outputWeights [[buffer(2)]],
                              device float *outputBiases [[buffer(3)]],
                              const device float *trainingInputs [[buffer(4)]],
                              const device float *expectedOutputs [[buffer(5)]],
                              constant uint &epochCount [[buffer(6)]],
                              constant uint &exampleCount [[buffer(7)]],
                              constant float &learningRate [[buffer(8)]],
                              uint threadIndex [[thread_position_in_grid]]) {
  if (threadIndex != 0)
    return;

  for (uint epochIndex = 0; epochIndex < epochCount; ++epochIndex) {
    float dHiddenWeights0 = 0.0f;
    float dHiddenWeights1 = 0.0f;
    float dHiddenWeights2 = 0.0f;
    float dHiddenWeights3 = 0.0f;
    float dHiddenBiases0 = 0.0f;
    float dHiddenBiases1 = 0.0f;
    float dOutputWeights0 = 0.0f;
    float dOutputWeights1 = 0.0f;
    float dOutputBias = 0.0f;

    for (uint exampleIndex = 0; exampleIndex < exampleCount; ++exampleIndex) {
      uint inputOffset = exampleIndex * 2;
      float input0 = trainingInputs[inputOffset];
      float input1 = trainingInputs[inputOffset + 1];

      float hidden0 = sigmoid(hiddenBiases[0] + (hiddenWeights[0] * input0) +
                              (hiddenWeights[1] * input1));
      float hidden1 = sigmoid(hiddenBiases[1] + (hiddenWeights[2] * input0) +
                              (hiddenWeights[3] * input1));
      float output = sigmoid(outputBiases[0] + (outputWeights[0] * hidden0) +
                             (outputWeights[1] * hidden1));
      float outputError = output - expectedOutputs[exampleIndex];

      dOutputWeights0 += outputError * hidden0;
      dOutputWeights1 += outputError * hidden1;
      dOutputBias += outputError;

      float hiddenError0 =
          outputWeights[0] * outputError * hidden0 * (1.0f - hidden0);
      float hiddenError1 =
          outputWeights[1] * outputError * hidden1 * (1.0f - hidden1);

      dHiddenWeights0 += hiddenError0 * input0;
      dHiddenWeights1 += hiddenError0 * input1;
      dHiddenWeights2 += hiddenError1 * input0;
      dHiddenWeights3 += hiddenError1 * input1;
      dHiddenBiases0 += hiddenError0;
      dHiddenBiases1 += hiddenError1;
    }

    hiddenWeights[0] -= learningRate * dHiddenWeights0;
    hiddenWeights[1] -= learningRate * dHiddenWeights1;
    hiddenWeights[2] -= learningRate * dHiddenWeights2;
    hiddenWeights[3] -= learningRate * dHiddenWeights3;
    hiddenBiases[0] -= learningRate * dHiddenBiases0;
    hiddenBiases[1] -= learningRate * dHiddenBiases1;
    outputWeights[0] -= learningRate * dOutputWeights0;
    outputWeights[1] -= learningRate * dOutputWeights1;
    outputBiases[0] -= learningRate * dOutputBias;
  }
}
