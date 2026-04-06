let assertPredictionClose evaluation =
  let predictionError =
    Float.abs
      (evaluation.Ocaml_machine_learning.XorTraining.predictedOutput
     -. evaluation.Ocaml_machine_learning.XorTraining.expectedOutput)
  in
  if predictionError > 0.15 then
    failwith
      (Printf.sprintf "Prediction %.4f was too far from expected %.1f"
         evaluation.Ocaml_machine_learning.XorTraining.predictedOutput
         evaluation.Ocaml_machine_learning.XorTraining.expectedOutput)

let () =
  let trainedNetwork =
    Ocaml_machine_learning.XorTraining.train
      ~epochCount:Ocaml_machine_learning.XorTraining.defaultEpochCount
      ~learningRate:Ocaml_machine_learning.XorTraining.defaultLearningRate
      ~randomSeed:Ocaml_machine_learning.XorTraining.defaultRandomSeed
  in
  Ocaml_machine_learning.XorTraining.evaluate trainedNetwork
  |> List.iter assertPredictionClose
