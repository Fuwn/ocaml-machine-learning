type t

val create :
  randomState:Random.State.t -> inputCount:int -> hiddenCount:int -> t

val predict : t -> float array -> float

val trainExample :
  t -> learningRate:float -> inputs:float array -> expectedOutput:float -> unit
