type evaluation = {
  inputs : float array;
  expectedOutput : float;
  predictedOutput : float;
}

val defaultEpochCount : int
val defaultLearningRate : float
val defaultRandomSeed : int

val train :
  epochCount:int ->
  learningRate:float ->
  randomSeed:int ->
  TwoLayerNeuralNetwork.t

val evaluate : TwoLayerNeuralNetwork.t -> evaluation list
