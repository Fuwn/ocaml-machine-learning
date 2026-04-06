type evaluation = {
  inputs : float array;
  expectedOutput : float;
  predictedOutput : float;
}

let defaultEpochCount = 40_000
let defaultLearningRate = 0.7
let defaultRandomSeed = 7
let defaultBackendKind = ComputeBackend.Cpu

let xorTrainingExamples =
  [
    TrainingExample.create [| 0.0; 0.0 |] 0.0;
    TrainingExample.create [| 0.0; 1.0 |] 1.0;
    TrainingExample.create [| 1.0; 0.0 |] 1.0;
    TrainingExample.create [| 1.0; 1.0 |] 0.0;
  ]

let trainingInputs =
  Array.of_list
    (List.map
       (fun (trainingExample : TrainingExample.t) ->
         Array.copy trainingExample.TrainingExample.inputs)
       xorTrainingExamples)

let expectedOutputs =
  Array.of_list
    (List.map
       (fun (trainingExample : TrainingExample.t) ->
         trainingExample.TrainingExample.expectedOutput)
       xorTrainingExamples)

let trainWithBackend ~backendKind ~epochCount ~learningRate ~randomSeed =
  let randomState = Random.State.make [| randomSeed |] in
  let network =
    TwoLayerNeuralNetwork.create ~backendKind ~randomState ~inputCount:2
      ~hiddenCount:2
  in
  TwoLayerNeuralNetwork.trainExamples network ~epochCount ~learningRate
    ~trainingInputs ~expectedOutputs;
  network

let train ~epochCount ~learningRate ~randomSeed =
  trainWithBackend ~backendKind:defaultBackendKind ~epochCount ~learningRate
    ~randomSeed

let evaluate network =
  let predictedOutputs =
    TwoLayerNeuralNetwork.predictBatch network trainingInputs
  in
  Array.to_list
    (Array.mapi
       (fun exampleIndex inputs ->
         {
           inputs = Array.copy inputs;
           expectedOutput = expectedOutputs.(exampleIndex);
           predictedOutput = predictedOutputs.(exampleIndex);
         })
       trainingInputs)
