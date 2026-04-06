type evaluation = {
  inputs : float array;
  expectedOutput : float;
  predictedOutput : float;
}

val defaultEpochCount : int
val defaultLearningRate : float
val defaultRandomSeed : int
val defaultBackendKind : ComputeBackend.kind

val trainWithBackend :
  backendKind:ComputeBackend.kind ->
  epochCount:int ->
  learningRate:float ->
  randomSeed:int ->
  TwoLayerNeuralNetwork.t

val train :
  epochCount:int ->
  learningRate:float ->
  randomSeed:int ->
  TwoLayerNeuralNetwork.t

val evaluate : TwoLayerNeuralNetwork.t -> evaluation list
