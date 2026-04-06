type kind = Cpu | AppleGpu
type vector = float array
type matrix = float array array

let createWeight randomState = Random.State.float randomState 2.0 -. 1.0

let createRandomMatrix ~randomState ~rowCount ~columnCount =
  Array.init rowCount (fun _ ->
      Array.init columnCount (fun _ -> createWeight randomState))

let createRandomVector ~randomState ~count =
  Array.init count (fun _ -> createWeight randomState)

let validateMatrixVectorDimensions weights inputs biases =
  let rowCount = Array.length weights in
  if rowCount <> Array.length biases then
    invalid_arg "Matrix row count must match bias count";
  Array.iter
    (fun weightRow ->
      if Array.length weightRow <> Array.length inputs then
        invalid_arg "Matrix column count must match input count")
    weights

let validateVectorPair left right =
  if Array.length left <> Array.length right then
    invalid_arg "Vector lengths must match"

let appleGpuUnavailable operationName =
  failwith
    (Printf.sprintf "Apple GPU backend does not implement %s at this layer"
       operationName)

let cpuAffineTransform ~weights ~inputs ~biases =
  validateMatrixVectorDimensions weights inputs biases;
  Array.mapi
    (fun rowIndex weightRow ->
      let weightedSum = ref biases.(rowIndex) in
      Array.iteri
        (fun inputIndex inputValue ->
          weightedSum := !weightedSum +. (weightRow.(inputIndex) *. inputValue))
        inputs;
      !weightedSum)
    weights

let cpuMap mapping values = Array.map mapping values

let cpuTransposeMatrixVectorProduct weights values =
  let rowCount = Array.length weights in
  if rowCount <> Array.length values then
    invalid_arg "Matrix row count must match vector length";
  if rowCount = 0 then [||]
  else
    let columnCount = Array.length weights.(0) in
    Array.iter
      (fun weightRow ->
        if Array.length weightRow <> columnCount then
          invalid_arg "Matrix rows must have the same length")
      weights;
    Array.init columnCount (fun columnIndex ->
        let weightedSum = ref 0.0 in
        Array.iteri
          (fun rowIndex weightRow ->
            weightedSum :=
              !weightedSum +. (weightRow.(columnIndex) *. values.(rowIndex)))
          weights;
        !weightedSum)

let cpuHadamardProduct ~left ~right =
  validateVectorPair left right;
  Array.mapi (fun index leftValue -> leftValue *. right.(index)) left

let cpuUpdateMatrixByOuterProduct matrix ~learningRate ~errorSignals ~inputs =
  let rowCount = Array.length matrix in
  if rowCount <> Array.length errorSignals then
    invalid_arg "Matrix row count must match error signal count";
  Array.iteri
    (fun rowIndex weightRow ->
      if Array.length weightRow <> Array.length inputs then
        invalid_arg "Matrix column count must match input count";
      Array.iteri
        (fun columnIndex inputValue ->
          weightRow.(columnIndex) <-
            weightRow.(columnIndex)
            -. (learningRate *. errorSignals.(rowIndex) *. inputValue))
        inputs)
    matrix

let cpuUpdateVectorByScaledErrorSignal vector ~learningRate ~errorSignals =
  validateVectorPair vector errorSignals;
  Array.iteri
    (fun index errorSignal ->
      vector.(index) <- vector.(index) -. (learningRate *. errorSignal))
    errorSignals

let affineTransform backendKind ~weights ~inputs ~biases =
  match backendKind with
  | Cpu -> cpuAffineTransform ~weights ~inputs ~biases
  | AppleGpu -> appleGpuUnavailable "affineTransform"

let mapSigmoid backendKind values =
  match backendKind with
  | Cpu -> cpuMap ActivationFunction.sigmoid values
  | AppleGpu -> appleGpuUnavailable "mapSigmoid"

let mapSigmoidDerivativeFromOutput backendKind values =
  match backendKind with
  | Cpu -> cpuMap ActivationFunction.sigmoidDerivativeFromOutput values
  | AppleGpu -> appleGpuUnavailable "mapSigmoidDerivativeFromOutput"

let transposeMatrixVectorProduct backendKind weights values =
  match backendKind with
  | Cpu -> cpuTransposeMatrixVectorProduct weights values
  | AppleGpu -> appleGpuUnavailable "transposeMatrixVectorProduct"

let hadamardProduct backendKind ~left ~right =
  match backendKind with
  | Cpu -> cpuHadamardProduct ~left ~right
  | AppleGpu -> appleGpuUnavailable "hadamardProduct"

let updateMatrixByOuterProduct backendKind matrix ~learningRate ~errorSignals
    ~inputs =
  match backendKind with
  | Cpu ->
      cpuUpdateMatrixByOuterProduct matrix ~learningRate ~errorSignals ~inputs
  | AppleGpu -> appleGpuUnavailable "updateMatrixByOuterProduct"

let updateVectorByScaledErrorSignal backendKind vector ~learningRate
    ~errorSignals =
  match backendKind with
  | Cpu -> cpuUpdateVectorByScaledErrorSignal vector ~learningRate ~errorSignals
  | AppleGpu -> appleGpuUnavailable "updateVectorByScaledErrorSignal"
