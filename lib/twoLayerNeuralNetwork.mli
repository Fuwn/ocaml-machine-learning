type t

val create :
  backendKind:ComputeBackend.kind ->
  randomState:Random.State.t ->
  inputCount:int ->
  hiddenCount:int ->
  t

val trainExamples :
  t ->
  epochCount:int ->
  learningRate:float ->
  trainingInputs:float array array ->
  expectedOutputs:float array ->
  unit

val predictBatch : t -> float array array -> float array
val predict : t -> float array -> float

val trainExample :
  t -> learningRate:float -> inputs:float array -> expectedOutput:float -> unit
