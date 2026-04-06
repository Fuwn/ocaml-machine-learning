type neuron = { inputWeights : float array; mutable bias : float }
type t = { inputCount : int; hiddenLayer : neuron array; outputNeuron : neuron }

let createWeight randomState = Random.State.float randomState 2.0 -. 1.0

let createNeuron randomState inputCount =
  let inputWeights =
    Array.init inputCount (fun _ -> createWeight randomState)
  in
  let bias = createWeight randomState in
  { inputWeights; bias }

let validateInputCount expectedInputCount inputs =
  let actualInputCount = Array.length inputs in
  if actualInputCount <> expectedInputCount then
    invalid_arg
      (Printf.sprintf "Expected %d inputs but received %d" expectedInputCount
         actualInputCount)

let neuronOutput neuron inputs =
  let weightedSum = ref neuron.bias in
  Array.iteri
    (fun inputIndex inputValue ->
      weightedSum :=
        !weightedSum +. (neuron.inputWeights.(inputIndex) *. inputValue))
    inputs;
  ActivationFunction.sigmoid !weightedSum

let hiddenOutputs network inputs =
  Array.map (fun neuron -> neuronOutput neuron inputs) network.hiddenLayer

let forwardPass network inputs =
  validateInputCount network.inputCount inputs;
  let hiddenLayerOutputs = hiddenOutputs network inputs in
  let output = neuronOutput network.outputNeuron hiddenLayerOutputs in
  (hiddenLayerOutputs, output)

let create ~randomState ~inputCount ~hiddenCount =
  if inputCount < 1 then invalid_arg "inputCount must be at least 1";
  if hiddenCount < 1 then invalid_arg "hiddenCount must be at least 1";
  let hiddenLayer =
    Array.init hiddenCount (fun _ -> createNeuron randomState inputCount)
  in
  let outputNeuron = createNeuron randomState hiddenCount in
  { inputCount; hiddenLayer; outputNeuron }

let predict network inputs =
  let _, output = forwardPass network inputs in
  output

let trainExample network ~learningRate ~inputs ~expectedOutput =
  let hiddenLayerOutputs, output = forwardPass network inputs in
  let outputWeightsBeforeUpdate =
    Array.copy network.outputNeuron.inputWeights
  in
  let outputErrorSignal =
    (output -. expectedOutput)
    *. ActivationFunction.sigmoidDerivativeFromOutput output
  in
  Array.iteri
    (fun hiddenNeuronIndex hiddenOutput ->
      let updatedWeight =
        network.outputNeuron.inputWeights.(hiddenNeuronIndex)
        -. (learningRate *. outputErrorSignal *. hiddenOutput)
      in
      network.outputNeuron.inputWeights.(hiddenNeuronIndex) <- updatedWeight)
    hiddenLayerOutputs;
  network.outputNeuron.bias <-
    network.outputNeuron.bias -. (learningRate *. outputErrorSignal);
  Array.iteri
    (fun hiddenNeuronIndex hiddenNeuron ->
      let hiddenOutput = hiddenLayerOutputs.(hiddenNeuronIndex) in
      let hiddenErrorSignal =
        outputWeightsBeforeUpdate.(hiddenNeuronIndex)
        *. outputErrorSignal
        *. ActivationFunction.sigmoidDerivativeFromOutput hiddenOutput
      in
      Array.iteri
        (fun inputIndex inputValue ->
          let updatedWeight =
            hiddenNeuron.inputWeights.(inputIndex)
            -. (learningRate *. hiddenErrorSignal *. inputValue)
          in
          hiddenNeuron.inputWeights.(inputIndex) <- updatedWeight)
        inputs;
      hiddenNeuron.bias <-
        hiddenNeuron.bias -. (learningRate *. hiddenErrorSignal))
    network.hiddenLayer
