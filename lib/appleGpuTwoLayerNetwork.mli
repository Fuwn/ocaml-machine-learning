val trainEpochs :
  epochCount:int ->
  learningRate:float ->
  hiddenWeights:float array array ->
  hiddenBiases:float array ->
  outputWeights:float array array ->
  outputBiases:float array ->
  trainingInputs:float array array ->
  expectedOutputs:float array ->
  unit

val predictBatch :
  hiddenWeights:float array array ->
  hiddenBiases:float array ->
  outputWeights:float array array ->
  outputBiases:float array ->
  inputsBatch:float array array ->
  float array
