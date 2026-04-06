type layer = { weights : ComputeBackend.matrix; biases : ComputeBackend.vector }

type t = {
  backendKind : ComputeBackend.kind;
  inputCount : int;
  hiddenLayer : layer;
  outputLayer : layer;
}

let validateInputCount expectedInputCount inputs =
  let actualInputCount = Array.length inputs in
  if actualInputCount <> expectedInputCount then
    invalid_arg
      (Printf.sprintf "Expected %d inputs but received %d" expectedInputCount
         actualInputCount)

let validateExampleInputs network inputsBatch =
  Array.iter (validateInputCount network.inputCount) inputsBatch

let validateExpectedOutputCount trainingInputs expectedOutputs =
  if Array.length trainingInputs <> Array.length expectedOutputs then
    invalid_arg "Training input count must match expected output count"

let forwardPass network inputs =
  validateInputCount network.inputCount inputs;
  let hiddenLayerOutputs =
    ComputeBackend.affineTransform network.backendKind
      ~weights:network.hiddenLayer.weights ~inputs
      ~biases:network.hiddenLayer.biases
    |> ComputeBackend.mapSigmoid network.backendKind
  in
  let output =
    ComputeBackend.affineTransform network.backendKind
      ~weights:network.outputLayer.weights ~inputs:hiddenLayerOutputs
      ~biases:network.outputLayer.biases
    |> fun outputVector -> ActivationFunction.sigmoid outputVector.(0)
  in
  (hiddenLayerOutputs, output)

let create ~backendKind ~randomState ~inputCount ~hiddenCount =
  if inputCount < 1 then invalid_arg "inputCount must be at least 1";
  if hiddenCount < 1 then invalid_arg "hiddenCount must be at least 1";
  let hiddenLayer =
    {
      weights =
        ComputeBackend.createRandomMatrix ~randomState ~rowCount:hiddenCount
          ~columnCount:inputCount;
      biases = ComputeBackend.createRandomVector ~randomState ~count:hiddenCount;
    }
  in
  let outputLayer =
    {
      weights =
        ComputeBackend.createRandomMatrix ~randomState ~rowCount:1
          ~columnCount:hiddenCount;
      biases = ComputeBackend.createRandomVector ~randomState ~count:1;
    }
  in
  { backendKind; inputCount; hiddenLayer; outputLayer }

let predictBatch network inputsBatch =
  validateExampleInputs network inputsBatch;
  match network.backendKind with
  | ComputeBackend.Cpu ->
      Array.map
        (fun inputs ->
          let _, output = forwardPass network inputs in
          output)
        inputsBatch
  | ComputeBackend.AppleGpu ->
      AppleGpuTwoLayerNetwork.predictBatch
        ~hiddenWeights:network.hiddenLayer.weights
        ~hiddenBiases:network.hiddenLayer.biases
        ~outputWeights:network.outputLayer.weights
        ~outputBiases:network.outputLayer.biases ~inputsBatch

let predict network inputs = (predictBatch network [| inputs |]).(0)

let trainExample network ~learningRate ~inputs ~expectedOutput =
  let hiddenLayerOutputs, output = forwardPass network inputs in
  let outputErrorSignal =
    [|
      (output -. expectedOutput)
      *. ActivationFunction.sigmoidDerivativeFromOutput output;
    |]
  in
  let hiddenErrorSignal =
    ComputeBackend.transposeMatrixVectorProduct network.backendKind
      network.outputLayer.weights outputErrorSignal
    |> fun propagatedOutputErrorSignal ->
    ComputeBackend.hadamardProduct network.backendKind
      ~left:propagatedOutputErrorSignal
      ~right:
        (ComputeBackend.mapSigmoidDerivativeFromOutput network.backendKind
           hiddenLayerOutputs)
  in
  ComputeBackend.updateMatrixByOuterProduct network.backendKind
    network.outputLayer.weights ~learningRate ~errorSignals:outputErrorSignal
    ~inputs:hiddenLayerOutputs;
  ComputeBackend.updateVectorByScaledErrorSignal network.backendKind
    network.outputLayer.biases ~learningRate ~errorSignals:outputErrorSignal;
  ComputeBackend.updateMatrixByOuterProduct network.backendKind
    network.hiddenLayer.weights ~learningRate ~errorSignals:hiddenErrorSignal
    ~inputs;
  ComputeBackend.updateVectorByScaledErrorSignal network.backendKind
    network.hiddenLayer.biases ~learningRate ~errorSignals:hiddenErrorSignal

let trainExamples network ~epochCount ~learningRate ~trainingInputs
    ~expectedOutputs =
  validateExampleInputs network trainingInputs;
  validateExpectedOutputCount trainingInputs expectedOutputs;
  match network.backendKind with
  | ComputeBackend.Cpu ->
      for _ = 1 to epochCount do
        Array.iteri
          (fun exampleIndex inputs ->
            trainExample network ~learningRate ~inputs
              ~expectedOutput:expectedOutputs.(exampleIndex))
          trainingInputs
      done
  | ComputeBackend.AppleGpu ->
      AppleGpuTwoLayerNetwork.trainEpochs ~epochCount ~learningRate
        ~hiddenWeights:network.hiddenLayer.weights
        ~hiddenBiases:network.hiddenLayer.biases
        ~outputWeights:network.outputLayer.weights
        ~outputBiases:network.outputLayer.biases ~trainingInputs
        ~expectedOutputs
