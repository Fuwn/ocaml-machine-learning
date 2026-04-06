type evaluation = {
  inputs : float array;
  expectedOutput : float;
  predictedOutput : float;
}

let defaultEpochCount = 40_000
let defaultLearningRate = 0.7
let defaultRandomSeed = 7

let xorTrainingExamples =
  [
    TrainingExample.create [| 0.0; 0.0 |] 0.0;
    TrainingExample.create [| 0.0; 1.0 |] 1.0;
    TrainingExample.create [| 1.0; 0.0 |] 1.0;
    TrainingExample.create [| 1.0; 1.0 |] 0.0;
  ]

let train ~epochCount ~learningRate ~randomSeed =
  let randomState = Random.State.make [| randomSeed |] in
  let network =
    TwoLayerNeuralNetwork.create ~randomState ~inputCount:2 ~hiddenCount:2
  in
  for _ = 1 to epochCount do
    List.iter
      (fun (trainingExample : TrainingExample.t) ->
        TwoLayerNeuralNetwork.trainExample network ~learningRate
          ~inputs:trainingExample.TrainingExample.inputs
          ~expectedOutput:trainingExample.TrainingExample.expectedOutput)
      xorTrainingExamples
  done;
  network

let evaluate network =
  List.map
    (fun (trainingExample : TrainingExample.t) ->
      let predictedOutput =
        TwoLayerNeuralNetwork.predict network
          trainingExample.TrainingExample.inputs
      in
      {
        inputs = Array.copy trainingExample.TrainingExample.inputs;
        expectedOutput = trainingExample.TrainingExample.expectedOutput;
        predictedOutput;
      })
    xorTrainingExamples
