type benchmarkConfiguration = {
  name : string;
  batchSize : int;
  repetitionCount : int;
}

type backendMeasurement = {
  backendName : string;
  trainingSeconds : float;
  inferenceResults : (string * float) list;
}

let trainingEpochCount = 10_000
let timingRunCount = 3

let inferenceConfigurations =
  [
    { name = "single_prediction"; batchSize = 1; repetitionCount = 50_000 };
    { name = "small_batch"; batchSize = 32; repetitionCount = 5_000 };
    { name = "large_batch"; batchSize = 10_000; repetitionCount = 50 };
  ]

let currentTimeInSeconds () =
  Unix.gettimeofday ()

let median values =
  let sortedValues = List.sort Float.compare values in
  let valueCount = List.length sortedValues in
  let middleIndex = valueCount / 2 in
  if valueCount mod 2 = 1 then
    List.nth sortedValues middleIndex
  else
    let lowerValue = List.nth sortedValues (middleIndex - 1) in
    let upperValue = List.nth sortedValues middleIndex in
    (lowerValue +. upperValue) /. 2.0

let measureSeconds operation =
  let startedAt = currentTimeInSeconds () in
  let result = operation () in
  let finishedAt = currentTimeInSeconds () in
  (result, finishedAt -. startedAt)

let repeatMeasurement ~runCount operation =
  let rec collectMeasurements remainingRunCount accumulatedValues =
    if remainingRunCount = 0 then
      List.rev accumulatedValues
    else
      let _, elapsedSeconds = measureSeconds operation in
      collectMeasurements (remainingRunCount - 1)
        (elapsedSeconds :: accumulatedValues)
  in
  collectMeasurements runCount []

let backendName = function
  | Ocaml_machine_learning.ComputeBackend.Cpu -> "cpu"
  | Ocaml_machine_learning.ComputeBackend.AppleGpu -> "apple_gpu"

let trainNetwork backendKind epochCount =
  Ocaml_machine_learning.XorTraining.trainWithBackend ~backendKind
    ~epochCount
    ~learningRate:Ocaml_machine_learning.XorTraining.defaultLearningRate
    ~randomSeed:Ocaml_machine_learning.XorTraining.defaultRandomSeed

let makeInputsBatch batchSize =
  Array.init batchSize (fun _ -> [| 0.0; 1.0 |])

let runInference network configuration =
  let inputsBatch = makeInputsBatch configuration.batchSize in
  let predictionAccumulator = ref 0.0 in
  for _ = 1 to configuration.repetitionCount do
    let predictions =
      Ocaml_machine_learning.TwoLayerNeuralNetwork.predictBatch network
        inputsBatch
    in
    predictionAccumulator :=
      !predictionAccumulator
      +. Array.fold_left ( +. ) 0.0 predictions
  done;
  ignore (Sys.opaque_identity !predictionAccumulator)

let measureBackend backendKind =
  let networkWarmup = trainNetwork backendKind 1_000 in
  ignore
    (Ocaml_machine_learning.TwoLayerNeuralNetwork.predictBatch networkWarmup
       (makeInputsBatch 32));
  let trainingSeconds =
    repeatMeasurement ~runCount:timingRunCount (fun () ->
        ignore (trainNetwork backendKind trainingEpochCount))
    |> median
  in
  let inferenceNetwork = trainNetwork backendKind trainingEpochCount in
  let inferenceResults =
    List.map
      (fun configuration ->
        let elapsedSeconds =
          repeatMeasurement ~runCount:timingRunCount (fun () ->
              runInference inferenceNetwork configuration)
          |> median
        in
        (configuration.name, elapsedSeconds))
      inferenceConfigurations
  in
  { backendName = backendName backendKind; trainingSeconds; inferenceResults }

let printMeasurement measurement =
  Printf.printf "backend=%s training_epochs=%d median_training_seconds=%.6f\n"
    measurement.backendName
    trainingEpochCount
    measurement.trainingSeconds;
  List.iter
    (fun (name, elapsedSeconds) ->
      Printf.printf "backend=%s benchmark=%s median_seconds=%.6f\n"
        measurement.backendName
        name
        elapsedSeconds)
    measurement.inferenceResults

let () =
  [
    Ocaml_machine_learning.ComputeBackend.Cpu;
    Ocaml_machine_learning.ComputeBackend.AppleGpu;
  ]
  |> List.map measureBackend
  |> List.iter printMeasurement
