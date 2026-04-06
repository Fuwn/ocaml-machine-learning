external trainEpochsFlat :
  int ->
  float ->
  float array ->
  float array ->
  float array ->
  float array ->
  float array ->
  float array ->
  unit = "caml_apple_gpu_train_epochs_bytecode" "caml_apple_gpu_train_epochs"

external predictBatchFlat :
  float array ->
  float array ->
  float array ->
  float array ->
  float array ->
  int ->
  float array
  = "caml_apple_gpu_predict_batch_bytecode" "caml_apple_gpu_predict_batch"

let flattenMatrix matrix = Array.concat (Array.to_list matrix)

let copyFlatMatrixIntoMatrix flatValues matrix =
  let columnCount = Array.length matrix.(0) in
  Array.iteri
    (fun rowIndex row ->
      Array.blit flatValues (rowIndex * columnCount) row 0 columnCount)
    matrix

let trainEpochs ~epochCount ~learningRate ~hiddenWeights ~hiddenBiases
    ~outputWeights ~outputBiases ~trainingInputs ~expectedOutputs =
  let hiddenWeightsFlat = flattenMatrix hiddenWeights in
  let outputWeightsFlat = flattenMatrix outputWeights in
  let trainingInputsFlat = flattenMatrix trainingInputs in
  trainEpochsFlat epochCount learningRate hiddenWeightsFlat hiddenBiases
    outputWeightsFlat outputBiases trainingInputsFlat expectedOutputs;
  copyFlatMatrixIntoMatrix hiddenWeightsFlat hiddenWeights;
  copyFlatMatrixIntoMatrix outputWeightsFlat outputWeights

let predictBatch ~hiddenWeights ~hiddenBiases ~outputWeights ~outputBiases
    ~inputsBatch =
  let hiddenWeightsFlat = flattenMatrix hiddenWeights in
  let outputWeightsFlat = flattenMatrix outputWeights in
  let inputsBatchFlat = flattenMatrix inputsBatch in
  predictBatchFlat hiddenWeightsFlat hiddenBiases outputWeightsFlat outputBiases
    inputsBatchFlat (Array.length inputsBatch)
