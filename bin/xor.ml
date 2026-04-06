let printEvaluation evaluation =
  Printf.printf "[%.0f, %.0f] -> expected %.0f, predicted %.4f\n"
    evaluation.Ocaml_machine_learning.XorTraining.inputs.(0)
    evaluation.Ocaml_machine_learning.XorTraining.inputs.(1)
    evaluation.Ocaml_machine_learning.XorTraining.expectedOutput
    evaluation.Ocaml_machine_learning.XorTraining.predictedOutput

let () =
  let trainedNetwork =
    Ocaml_machine_learning.XorTraining.train
      ~epochCount:Ocaml_machine_learning.XorTraining.defaultEpochCount
      ~learningRate:Ocaml_machine_learning.XorTraining.defaultLearningRate
      ~randomSeed:Ocaml_machine_learning.XorTraining.defaultRandomSeed
  in
  let evaluations =
    Ocaml_machine_learning.XorTraining.evaluate trainedNetwork
  in
  List.iter printEvaluation evaluations
