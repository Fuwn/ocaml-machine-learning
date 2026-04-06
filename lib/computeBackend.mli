type kind = Cpu | AppleGpu
type vector = float array
type matrix = float array array

val createRandomMatrix :
  randomState:Random.State.t -> rowCount:int -> columnCount:int -> matrix

val createRandomVector : randomState:Random.State.t -> count:int -> vector

val affineTransform :
  kind -> weights:matrix -> inputs:vector -> biases:vector -> vector

val mapSigmoid : kind -> vector -> vector
val mapSigmoidDerivativeFromOutput : kind -> vector -> vector
val transposeMatrixVectorProduct : kind -> matrix -> vector -> vector
val hadamardProduct : kind -> left:vector -> right:vector -> vector

val updateMatrixByOuterProduct :
  kind ->
  matrix ->
  learningRate:float ->
  errorSignals:vector ->
  inputs:vector ->
  unit

val updateVectorByScaledErrorSignal :
  kind -> vector -> learningRate:float -> errorSignals:vector -> unit
