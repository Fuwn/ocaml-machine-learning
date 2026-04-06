type t = { inputs : float array; expectedOutput : float }

let create inputs expectedOutput =
  { inputs = Array.copy inputs; expectedOutput }
