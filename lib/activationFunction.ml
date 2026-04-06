let sigmoid value = 1.0 /. (1.0 +. Float.exp (-.value))
let sigmoidDerivativeFromOutput output = output *. (1.0 -. output)
