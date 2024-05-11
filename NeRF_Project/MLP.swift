//
//  MLP.swift
//  mpsgraph_test
//
//  Created by Josh Lakin on 3/16/24.
//

import Foundation
import Metal
import MetalPerformanceShadersGraph

enum ActivationFunction {
    case ReLU
    case Sigmoid
    case None
}

class MLP {
    //var inputTensorData: [MPSGraphTensorData] = Array(repeating: MPSGraphTensorData(), count: 2)
    //var labelTensorData: [MPSGraphTensorData] = Array(repeating: MPSGraphTensorData(), count: 2)
//    var epochTensorData: [MPSGraphTensorData] = Array(repeating: MPSGraphTensorData(), count: 2)
    var layers: [Layer] = []
    var layerActivationFunctions: [ActivationFunction] = []
    var layerWeightTensors: [MPSGraphTensor] = []
    var layerBiasTensors: [MPSGraphTensor] = []
    var variableTensors: [MPSGraphTensor] = []
//    var momentumTensors: [MPSGraphTensor:MPSGraphTensor] = [:]
//    var velocityTensors: [MPSGraphTensor:MPSGraphTensor] = [:]
    var device: MTLDevice?
    var commandQueue: MTLCommandQueue?
    var doubleBufferingSemaphore: DispatchSemaphore = DispatchSemaphore(value: 2)
    var inputShape: [NSNumber]
    var labelSize: Int
    var graph: MPSGraph
    var lossTensor: MPSGraphTensor?
    var inputsPlaceholderTensor: MPSGraphTensor
    var labelsPlaceholderTensor: MPSGraphTensor
//    var epochPlaceholderTensor: MPSGraphTensor
    var targetTrainingTensors: [MPSGraphTensor] = []
    var targetInferenceTensor: MPSGraphTensor?
    var targetTrainingOps: [MPSGraphOperation] = []
    var loss: Float = 0.0
    init(inputShape: [NSNumber], labelSize: Int) {
        self.inputShape = inputShape
        self.labelSize = labelSize
        device = MTLCreateSystemDefaultDevice()
        if (device == nil) {
            fatalError("Could not instantiate Metal device")
        }
        commandQueue = device!.makeCommandQueue()
        if (commandQueue == nil) {
            fatalError("Could not instantiate Metal command queue")
        }
        graph = MPSGraph()
        inputsPlaceholderTensor = graph.placeholder(shape: inputShape, dataType: .float32, name: nil)
        labelsPlaceholderTensor = graph.placeholder(shape: [-1, labelSize as NSNumber], dataType: .float32, name: nil)
//        epochPlaceholderTensor = graph.placeholder(shape: [1], dataType: .float32, name: nil)
    }
    func lossFunction(tensors: [MPSGraphTensor], labelTensor: MPSGraphTensor) -> MPSGraphTensor {
        //square loss
        assert(tensors.count == 1, "tensors needs just activation of last layer")
        let lossSubtractTensor = graph.subtraction(tensors[0], labelTensor, name: nil)
        let lossSquareTensor = graph.multiplication(lossSubtractTensor, lossSubtractTensor, name: nil)
        let lossTensorSum = graph.reductionSum(with: lossSquareTensor, axis: 0, name: nil)
        let lossTensor = graph.reductionSum(with: lossTensorSum, axis: 1, name: nil)
        return lossTensor
    }
    func addDenseLayer(inputSize: Int, outputSize: Int, activationFunction: ActivationFunction, initialWeightArray: [Float]?, initialBiasArray: [Float]?) {
        let layer = Layer(numInputs: inputSize, numOutputs: outputSize)
        var weightTensor: MPSGraphTensor
        if (initialWeightArray == nil) {
            weightTensor = graph.variable(with: layer.weightData, shape: [inputSize as NSNumber, outputSize as NSNumber], dataType: .float32, name: nil)
        } else {
            var initial = initialWeightArray!
            let data = Data(bytes: &initial, count: MemoryLayout<Float>.size*initial.count)
            weightTensor = graph.variable(with: data, shape: [inputSize as NSNumber, outputSize as NSNumber], dataType: .float32, name: nil)
        }
//        let weightMomentumTensor = graph.variable(with: layer.weightData, shape: [inputSize as NSNumber, outputSize as NSNumber], dataType: .float32, name: nil)
//        let weightVelocityTensor = graph.variable(with: layer.weightData, shape: [inputSize as NSNumber, outputSize as NSNumber], dataType: .float32, name: nil)
        var biasTensor: MPSGraphTensor
        if (initialBiasArray == nil) {
            biasTensor = graph.variable(with: layer.biasData, shape: [outputSize as NSNumber], dataType: .float32, name: nil)
        } else {
            var initial = initialBiasArray!
            let data = Data(bytes: &initial, count: MemoryLayout<Float>.size*initial.count)
            biasTensor = graph.variable(with: data, shape: [outputSize as NSNumber], dataType: .float32, name: nil)
        }
//        let biasMomentumTensor = graph.variable(with: layer.biasData, shape: [outputSize as NSNumber], dataType: .float32, name: nil)
//        let biasVelocityTensor = graph.variable(with: layer.biasData, shape: [outputSize as NSNumber], dataType: .float32, name: nil)
        layerWeightTensors.append(weightTensor)
        variableTensors.append(weightTensor)
        layerBiasTensors.append(biasTensor)
        variableTensors.append(biasTensor)
//        momentumTensors[weightTensor] = weightMomentumTensor
//        momentumTensors[biasTensor] = biasMomentumTensor
//        velocityTensors[weightTensor] = weightVelocityTensor
//        velocityTensors[biasTensor] = biasVelocityTensor
        layers.append(layer)
        layerActivationFunctions.append(activationFunction)
    }
    func invokeMLP(inputTensor: MPSGraphTensor?) -> MPSGraphTensor {
        var activationTensor = inputsPlaceholderTensor
        if (inputTensor != nil) {
            activationTensor = inputTensor!
        }
        for i in 0..<layers.count {
            activationTensor = graph.matrixMultiplication(primary: activationTensor, secondary: layerWeightTensors[i], name: nil)
            activationTensor = graph.addition(activationTensor, layerBiasTensors[i], name: nil)
            if (layerActivationFunctions[i] == .Sigmoid) {
                activationTensor = graph.sigmoid(with: activationTensor, name: nil)
            } else if (layerActivationFunctions[i] == .ReLU) {
                activationTensor = graph.reLU(with: activationTensor, name: nil)
            }
        }
        return activationTensor
    }
    func finalize(inputTensor: MPSGraphTensor?) {
        if (inputTensor != nil) {
            targetInferenceTensor = invokeMLP(inputTensor: inputTensor!)
        } else {
            targetInferenceTensor = invokeMLP(inputTensor: inputsPlaceholderTensor)
        }
        let lossTensor = lossFunction(tensors: [targetInferenceTensor!], labelTensor: labelsPlaceholderTensor)
        targetTrainingTensors.append(lossTensor)
        self.lossTensor = lossTensor
        let gradientTensors = graph.gradients(of: lossTensor, with: variableTensors, name: nil)
//        let beta1Tensor = graph.constant(0.9, dataType: .float32)
//        let beta2Tensor = graph.constant(0.99, dataType: .float32)
//        let epsilonTensor = graph.constant(1e-8, dataType: .float32)
        for variableTensor in variableTensors {
            let updateTensor = graph.stochasticGradientDescent(learningRate: graph.constant(0.001, dataType: .float32), values: variableTensor, gradient: gradientTensors[variableTensor]!, name: nil) //stochastic gradient descent for now, change to adam later
            let assignOp = graph.assign(variableTensor, tensor: updateTensor, name: nil)
//            let adamUpdate = graph.adam(learningRate: graph.constant(0.001, dataType: .float32), beta1: beta1Tensor, beta2: beta2Tensor, epsilon: epsilonTensor, beta1Power: graph.power(beta1Tensor, epochPlaceholderTensor, name: nil), beta2Power: graph.power(beta2Tensor, epochPlaceholderTensor, name: nil), values: variableTensor, momentum: momentumTensors[variableTensor]!, velocity: velocityTensors[variableTensor]!, maximumVelocity: nil, gradient: gradientTensors[variableTensor]!, name: nil)
//            let assignOp = graph.assign(variableTensor, tensor: adamUpdate[0], name: nil)
//            let momentumAssignOp = graph.assign(momentumTensors[variableTensor]!, tensor: adamUpdate[1], name: nil)
//            let velocityAssignOp = graph.assign(velocityTensors[variableTensor]!, tensor: adamUpdate[2], name: nil)
            targetTrainingOps.append(assignOp)
//            targetTrainingOps.append(momentumAssignOp)
//            targetTrainingOps.append(velocityAssignOp)
        }
    }
    func trainDoubleBuffered(inputData: [[Float]], labelData: [[Float]], numLabels: Int, numEpochs: Int, batchSize: Int, inputDataType: MPSDataType) {
        assert(inputData.count == labelData.count)
        var shuffledIndices = inputData.indices.shuffled()
        var inputDataShuffled = shuffledIndices.map{inputData[$0]}
        var labelDataShuffled = shuffledIndices.map{labelData[$0]}
        
        var inputBatch: [Float] = []
        var labelBatch: [Float] = []
        for i in 1...numEpochs {
            loss = 0.0
            var firstIndex = 0
            autoreleasepool {
                while (firstIndex < numLabels) {
                    inputBatch.removeAll()
                    labelBatch.removeAll()
                    let curBatchSize: Int = min(batchSize, numLabels-firstIndex)
                    for arr in inputDataShuffled[firstIndex..<firstIndex+curBatchSize] {
                        for elem in arr {
                            inputBatch.append(elem)
                        }
                    }
                    for arr in labelDataShuffled[firstIndex..<firstIndex+curBatchSize] {
                        for elem in arr {
                            labelBatch.append(elem)
                        }
                    }
                    
                    doubleBufferingSemaphore.wait()
                    let inputTensorDataTemp = Data(bytes: &inputBatch, count: MemoryLayout<Float>.size*inputBatch.count)
                    let inputTensorDataCurrent = MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device!), data: inputTensorDataTemp, shape: [curBatchSize as NSNumber, inputData[0].count as NSNumber], dataType: inputDataType)
                    //print(inputTensorData[currentTensorDataIndex].shape)
                    let labelTensorDataTemp = Data(bytes: &labelBatch, count: MemoryLayout<Float>.size*labelBatch.count)
                    let labelTensorDataCurrent = MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device!), data: labelTensorDataTemp, shape: [curBatchSize as NSNumber, labelData[0].count as NSNumber], dataType: .float32)
    //                var epoch = Float(i)
    //                let epochTensorDataTemp = Data(bytes: &epoch, count: MemoryLayout<Float>.size)
    //                epochTensorData[currentTensorDataIndex] = MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device!), data: epochTensorDataTemp, shape: [1], dataType: .float32)
                    //print(labelTensorData[currentTensorDataIndex].shape)
                    let execDescriptor = MPSGraphExecutionDescriptor()
                    execDescriptor.completionHandler = { (resultsDictionary, nil) in
                        self.doubleBufferingSemaphore.signal()
                        if (self.lossTensor != nil) {
                            let result = resultsDictionary[self.lossTensor!]
                            var loss: [Float] = Array(repeating: 0.0, count: 1)
                            result!.mpsndarray().readBytes(&loss, strideBytes: nil)
                            self.loss += loss[0]
                        }
                    }
                    graph.runAsync(with: commandQueue!, feeds: [inputsPlaceholderTensor: inputTensorDataCurrent, labelsPlaceholderTensor: labelTensorDataCurrent], targetTensors: targetTrainingTensors, targetOperations: targetTrainingOps, executionDescriptor: execDescriptor)
    //                graph.runAsync(with: commandQueue!, feeds: [inputsPlaceholderTensor: inputTensorData[currentTensorDataIndex], labelsPlaceholderTensor: labelTensorData[currentTensorDataIndex], epochPlaceholderTensor: epochTensorData[currentTensorDataIndex]], targetTensors: targetTrainingTensors, targetOperations: targetTrainingOps, executionDescriptor: execDescriptor)
                    
                    firstIndex += batchSize
                }
            }

            shuffledIndices.shuffle()
            inputDataShuffled = shuffledIndices.map{inputData[$0]}
            labelDataShuffled = shuffledIndices.map{labelData[$0]}
            print("Loss: \(loss)")
        }
            
    }
    func getInference(inputData: [[Float]], getWeights: Bool, resultsOut: inout [MPSGraphTensor: MPSGraphTensorData]) -> [Float] {
        var results: [Float] = []
        if (targetInferenceTensor != nil) {
            var inputDataFlattened: [Float] = []
            for arr in inputData {
                for elem in arr {
                    inputDataFlattened.append(elem)
                }
            }
            let inputTensorData = Data(bytes: &inputDataFlattened, count: MemoryLayout<Float>.size*inputDataFlattened.count)
            let inputGraphTensorData = MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device!), data: inputTensorData, shape: [inputData.count as NSNumber, inputData[0].count as NSNumber], dataType: .float32)
            var targetTensors = [targetInferenceTensor!]
            if (getWeights) {
                targetTensors.append(contentsOf: variableTensors)
            }
            let resultsDict = graph.run(feeds: [inputsPlaceholderTensor : inputGraphTensorData], targetTensors: targetTensors, targetOperations: nil)
            resultsOut = resultsDict
            let resultsData = resultsDict[targetInferenceTensor!]
            if (resultsData != nil) {
                results = Array(repeating: 0.0, count: inputData.count*labelSize)
                resultsData!.mpsndarray().readBytes(&results, strideBytes: nil)
            }
            
        }
        return results
    }
    func getVariableData(results: [MPSGraphTensor: MPSGraphTensorData]) -> Data {
        var weights = CodableWeights()
        for i in 0..<layerWeightTensors.count {
            var weightMatrix: [Float] = Array(repeating: 0.0, count: layers[i].numInputs*layers[i].numOutputs)
            let tensorData: MPSGraphTensorData = results[layerWeightTensors[i]]!
            tensorData.mpsndarray().readBytes(&weightMatrix, strideBytes: nil)
            weights.weightMatrices.append(weightMatrix)
        }
        for i in 0..<layerBiasTensors.count {
            var biasVector: [Float] = Array(repeating: 0.0, count: layers[i].numOutputs)
            let tensorData: MPSGraphTensorData = results[layerBiasTensors[i]]!
            tensorData.mpsndarray().readBytes(&biasVector, strideBytes: nil)
            weights.biasVectors.append(biasVector)
        }
        let jsonEncoder = JSONEncoder()
        let jsonData = try! jsonEncoder.encode(weights)
        return jsonData
    }
}

let numLevels = 16
var prime1 = 2654435761
var prime2 = 805459861

enum Corner2D {
    case ll
    case lr
    case ul
    case ur
}

enum InterpolationMode {
    case linear
    case nearest
}

class MLP2D : MLP {
    var resolutionTensors: [MPSGraphTensor] = []
    var hashTableSizeTensors: [MPSGraphTensor] = []
    var hashTableWeightTensors: [MPSGraphTensor] = []
    var hashTableInitialized: Bool = false
    private func hashTensor(indexTensor: MPSGraphTensor, primeTensor: MPSGraphTensor, hashTableSizeTensor: MPSGraphTensor, corner: Corner2D) -> MPSGraphTensor {
        var xIndexTensor = graph.sliceTensor(indexTensor, dimension: 1, start: 0, length: 1, name: nil)
        var oneTensor = graph.constant(1, dataType: .float32)
        oneTensor = graph.cast(oneTensor, to: .int32, name: nil)
        if (corner == .lr || corner == .ur) {
            xIndexTensor = graph.addition(oneTensor, xIndexTensor, name: nil)
        }
        var yIndexTensor = graph.sliceTensor(indexTensor, dimension: 1, start: 1, length: 1, name: nil)
        if (corner == .ul || corner == .ur) {
            yIndexTensor = graph.addition(oneTensor, yIndexTensor, name: nil)
        }
        let yScaledTensor = graph.multiplication(yIndexTensor, primeTensor, name: nil)
        let xorTensor = graph.bitwiseXOR(xIndexTensor, yScaledTensor, name: nil)
        let moduloTensor = graph.modulo(xorTensor, hashTableSizeTensor, name: nil)
        return moduloTensor
    }
    func invokeHashTable() -> MPSGraphTensor {
        assert(hashTableInitialized)
        let prime1Data = Data(bytes: &prime1, count: MemoryLayout<Int>.size)
        let prime1Tensor = graph.constant(prime1Data, shape: [1], dataType: .int32)
        var interpolatedWeightTensors: [MPSGraphTensor] = []
        for i in 0..<numLevels {
            let resolutionFloatTensor = graph.cast(resolutionTensors[i], to: .float32, name: nil)
            let scaledInputTensor = graph.multiplication(resolutionFloatTensor, inputsPlaceholderTensor, name: nil)
            let lowerLeftTensor = graph.floor(with: scaledInputTensor, name: nil)
            let lowerLeftIndexTensor = graph.cast(lowerLeftTensor, to: .int32, name: nil)
            let lowerLeftHashTensor = hashTensor(indexTensor: lowerLeftIndexTensor, primeTensor: prime1Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .ll)
            var lowerLeftWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: lowerLeftHashTensor, batchDimensions: 0, name: nil)
            //print("Lower left weight Shape: \(lowerLeftWeightTensor.shape![0]) \(lowerLeftWeightTensor.shape![1])")
            let lowerRightHashTensor = hashTensor(indexTensor: lowerLeftIndexTensor, primeTensor: prime1Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .lr)
            let upperLeftHashTensor = hashTensor(indexTensor: lowerLeftIndexTensor, primeTensor: prime1Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .ul)
            let upperRightHashTensor = hashTensor(indexTensor: lowerLeftIndexTensor, primeTensor: prime1Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .ur)
            var lowerRightWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: lowerRightHashTensor, batchDimensions: 0, name: nil)
            var upperLeftWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: upperLeftHashTensor, batchDimensions: 0, name: nil)
            var upperRightWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: upperRightHashTensor, batchDimensions: 0, name: nil)
            let onesTensor = graph.constant(1.0, dataType: .float32)
            let lowerLeftDiffTensor = graph.subtraction(scaledInputTensor, lowerLeftTensor, name: nil)
            //print("Lower left diff Shape: \(lowerLeftDiffTensor.shape![0]) \(lowerLeftDiffTensor.shape![1])")
            let oneMinusLowerLeftDiffTensor = graph.subtraction(onesTensor, lowerLeftDiffTensor, name: nil)
            let lowerLeftDiffXTensor = graph.sliceTensor(lowerLeftDiffTensor, dimension: 1, start: 0, length: 1, name: nil)
            //print("Lower left diff X Shape: \(lowerLeftDiffXTensor.shape![0]) \(lowerLeftDiffXTensor.shape![1])")
            let lowerLeftDiffYTensor = graph.sliceTensor(lowerLeftDiffTensor, dimension: 1, start: 1, length: 1, name: nil)
            let oneMinusLowerLeftDiffXTensor = graph.sliceTensor(oneMinusLowerLeftDiffTensor, dimension: 1, start: 0, length: 1, name: nil)
            let oneMinusLowerLeftDiffYTensor = graph.sliceTensor(oneMinusLowerLeftDiffTensor, dimension: 1, start: 1, length: 1, name: nil)
            let upperRightScaleTensor = graph.multiplication(lowerLeftDiffXTensor, lowerLeftDiffYTensor, name: nil)
            let upperLeftScaleTensor = graph.multiplication(oneMinusLowerLeftDiffXTensor, lowerLeftDiffYTensor, name: nil)
            let lowerLeftScaleTensor = graph.multiplication(oneMinusLowerLeftDiffXTensor, oneMinusLowerLeftDiffYTensor, name: nil)
            let lowerRightScaleTensor = graph.multiplication(lowerLeftDiffXTensor, oneMinusLowerLeftDiffYTensor, name: nil)
            lowerLeftWeightTensor = graph.multiplication(lowerLeftScaleTensor, lowerLeftWeightTensor, name: nil)
            lowerRightWeightTensor = graph.multiplication(lowerRightScaleTensor, lowerRightWeightTensor, name: nil)
            upperLeftWeightTensor = graph.multiplication(upperLeftScaleTensor, upperLeftWeightTensor, name: nil)
            upperRightWeightTensor = graph.multiplication(upperRightScaleTensor, upperRightWeightTensor, name: nil)
            var interpolatedWeightTensor = graph.addition(lowerLeftWeightTensor, lowerRightWeightTensor, name: nil)
            interpolatedWeightTensor = graph.addition(interpolatedWeightTensor, upperLeftWeightTensor, name: nil)
            interpolatedWeightTensor = graph.addition(interpolatedWeightTensor, upperRightWeightTensor, name: nil)
            interpolatedWeightTensors.append(interpolatedWeightTensor)
            
        }
        let activationTensor = graph.concatTensors(interpolatedWeightTensors, dimension: 1, name: nil)
        //print("Activation Shape: \(activationTensor.shape![0]) \(activationTensor.shape![1])")
        return activationTensor
    }
    func createHashTable(minResolution: Int, maxResolution: Int, maxEntries: Int, initialHashTableWeights: [[Float]]?) {
        let b = exp( (log(Double(maxResolution)) - log(Double(minResolution))) / Double(numLevels - 1))
        for i in 0..<numLevels {
            var resolution: Int = Int(floor(Double(minResolution)*pow(b, Double(i))))
            let resolutionData = Data(bytes: &resolution, count: MemoryLayout<Int>.size)
            let resolutionTensor = graph.constant(resolutionData, shape: [1], dataType: .int32)
            resolutionTensors.append(resolutionTensor)
            var hashTableSize = min(resolution*resolution, maxEntries)
            let hashTableSizeData = Data(bytes: &hashTableSize, count: MemoryLayout<Int>.size)
            let hashTableSizeTensor = graph.constant(hashTableSizeData, shape: [1], dataType: .int32)
            hashTableSizeTensors.append(hashTableSizeTensor)
            print("Level \(i): \(hashTableSize) elements")
            var weightData: Data
            if (initialHashTableWeights == nil) {
                var arr: [Float] = []
                for _ in 0..<hashTableSize {
                    arr.append(Float.random(in: -1e-4...1e-4))
                    arr.append(Float.random(in: -1e-4...1e-4))
                }
                weightData = Data(bytes: &arr, count: arr.count*MemoryLayout<Float>.size)
            } else {
                var weights = initialHashTableWeights![i]
                weightData = Data(bytes: &weights, count: weights.count*MemoryLayout<Float>.size)
            }
            let weightTensor = graph.variable(with: weightData, shape: [hashTableSize as NSNumber, 2 as NSNumber], dataType: .float32, name: nil)
//            let weightMomentumTensor = graph.variable(with: weightData, shape: [resolution as NSNumber, 2 as NSNumber], dataType: .float32, name: nil)
//            let weightVelocityTensor = graph.variable(with: weightData, shape: [resolution as NSNumber, 2 as NSNumber], dataType: .float32, name: nil)
            hashTableWeightTensors.append(weightTensor)
            variableTensors.append(weightTensor)
//            momentumTensors[weightTensor] = weightMomentumTensor
//            velocityTensors[weightTensor] = weightVelocityTensor
        }
        hashTableInitialized = true
    }
    override func getVariableData(results: [MPSGraphTensor: MPSGraphTensorData]) -> Data {
        var weights = CodableWeights()
        for i in 0..<layerWeightTensors.count {
            var weightMatrix: [Float] = Array(repeating: 0.0, count: layers[i].numInputs*layers[i].numOutputs)
            let tensorData: MPSGraphTensorData = results[layerWeightTensors[i]]!
            tensorData.mpsndarray().readBytes(&weightMatrix, strideBytes: nil)
            weights.weightMatrices.append(weightMatrix)
        }
        for i in 0..<layerBiasTensors.count {
            var biasVector: [Float] = Array(repeating: 0.0, count: layers[i].numOutputs)
            let tensorData: MPSGraphTensorData = results[layerBiasTensors[i]]!
            tensorData.mpsndarray().readBytes(&biasVector, strideBytes: nil)
            weights.biasVectors.append(biasVector)
        }
        for i in 0..<hashTableWeightTensors.count {
            var hashTableWeights: [Float] = Array(repeating: 0.0, count: hashTableWeightTensors[i].shape![0].intValue*hashTableWeightTensors[i].shape![1].intValue)
            let tensorData: MPSGraphTensorData = results[hashTableWeightTensors[i]]!
            tensorData.mpsndarray().readBytes(&hashTableWeights, strideBytes: nil)
            weights.hashTableWeights.append(hashTableWeights)
        }
        let jsonEncoder = JSONEncoder()
        let jsonData = try! jsonEncoder.encode(weights)
        return jsonData
    }
    override init(inputShape: [NSNumber], labelSize: Int) {
        super.init(inputShape: inputShape, labelSize: labelSize)
    }
}

enum Face {
    case back
    case front
}

class MLP3D : MLP2D {
    private func hashTensor(indexTensor: MPSGraphTensor, primeTensor1: MPSGraphTensor, primeTensor2: MPSGraphTensor, hashTableSizeTensor: MPSGraphTensor, corner: Corner2D, face: Face, dimension: Int) -> MPSGraphTensor {
        var xIndexTensor = graph.sliceTensor(indexTensor, dimension: dimension, start: 0, length: 1, name: nil)
        var oneTensor = graph.constant(1, dataType: .float32)
        oneTensor = graph.cast(oneTensor, to: .int32, name: nil)
        if (corner == .lr || corner == .ur) {
            xIndexTensor = graph.addition(oneTensor, xIndexTensor, name: nil)
        }
        var yIndexTensor = graph.sliceTensor(indexTensor, dimension: dimension, start: 1, length: 1, name: nil)
        if (corner == .ul || corner == .ur) {
            yIndexTensor = graph.addition(oneTensor, yIndexTensor, name: nil)
        }
        var zIndexTensor = graph.sliceTensor(indexTensor, dimension: dimension, start: 2, length: 1, name: nil)
        if (face == .front) {
            zIndexTensor = graph.addition(oneTensor, zIndexTensor, name: nil)
        }
        let yScaledTensor = graph.multiplication(yIndexTensor, primeTensor1, name: nil)
        let zScaledTensor = graph.multiplication(zIndexTensor, primeTensor2, name: nil)
        var xorTensor = graph.bitwiseXOR(xIndexTensor, yScaledTensor, name: nil)
        xorTensor = graph.bitwiseXOR(xorTensor, zScaledTensor, name: nil)
        let moduloTensor = graph.modulo(xorTensor, hashTableSizeTensor, name: nil)
        return moduloTensor
    }
    private func interpolate2D(lowerLeftDiffTensor: MPSGraphTensor, oneMinusLowerLeftDiffTensor: MPSGraphTensor, lowerLeftWeightTensor: MPSGraphTensor, lowerRightWeightTensor: MPSGraphTensor, upperLeftWeightTensor: MPSGraphTensor, upperRightWeightTensor: MPSGraphTensor, dimension: Int) -> MPSGraphTensor {
        let lowerLeftDiffXTensor = graph.sliceTensor(lowerLeftDiffTensor, dimension: dimension, start: 0, length: 1, name: nil)
        //print("Lower left diff X Shape: \(lowerLeftDiffXTensor.shape![0]) \(lowerLeftDiffXTensor.shape![1])")
        let lowerLeftDiffYTensor = graph.sliceTensor(lowerLeftDiffTensor, dimension: dimension, start: 1, length: 1, name: nil)
        let oneMinusLowerLeftDiffXTensor = graph.sliceTensor(oneMinusLowerLeftDiffTensor, dimension: dimension, start: 0, length: 1, name: nil)
        let oneMinusLowerLeftDiffYTensor = graph.sliceTensor(oneMinusLowerLeftDiffTensor, dimension: dimension, start: 1, length: 1, name: nil)
        let upperRightScaleTensor = graph.multiplication(lowerLeftDiffXTensor, lowerLeftDiffYTensor, name: nil)
        let upperLeftScaleTensor = graph.multiplication(oneMinusLowerLeftDiffXTensor, lowerLeftDiffYTensor, name: nil)
        let lowerLeftScaleTensor = graph.multiplication(oneMinusLowerLeftDiffXTensor, oneMinusLowerLeftDiffYTensor, name: nil)
        let lowerRightScaleTensor = graph.multiplication(lowerLeftDiffXTensor, oneMinusLowerLeftDiffYTensor, name: nil)
        let lowerLeftWeightScaledTensor = graph.multiplication(lowerLeftScaleTensor, lowerLeftWeightTensor, name: nil)
        let lowerRightWeightScaledTensor = graph.multiplication(lowerRightScaleTensor, lowerRightWeightTensor, name: nil)
        let upperLeftWeightScaledTensor = graph.multiplication(upperLeftScaleTensor, upperLeftWeightTensor, name: nil)
        let upperRightWeightScaledTensor = graph.multiplication(upperRightScaleTensor, upperRightWeightTensor, name: nil)
        var interpolatedWeightTensor = graph.addition(lowerLeftWeightScaledTensor, lowerRightWeightScaledTensor, name: nil)
        interpolatedWeightTensor = graph.addition(interpolatedWeightTensor, upperLeftWeightScaledTensor, name: nil)
        interpolatedWeightTensor = graph.addition(interpolatedWeightTensor, upperRightWeightScaledTensor, name: nil)
        return interpolatedWeightTensor
    }
    override init(inputShape: [NSNumber], labelSize: Int) {
        super.init(inputShape: inputShape, labelSize: labelSize)
    }
    override func createHashTable(minResolution: Int, maxResolution: Int, maxEntries: Int, initialHashTableWeights: [[Float]]?) {
        let b = exp( (log(Double(maxResolution)) - log(Double(minResolution))) / Double(numLevels - 1))
        for i in 0..<numLevels {
            var resolution: Int = Int(floor(Double(minResolution)*pow(b, Double(i))))
            let resolutionData = Data(bytes: &resolution, count: MemoryLayout<Int>.size)
            let resolutionTensor = graph.constant(resolutionData, shape: [1], dataType: .int32)
            resolutionTensors.append(resolutionTensor)
            var hashTableSize = min(resolution*resolution*resolution, maxEntries)
            let hashTableSizeData = Data(bytes: &hashTableSize, count: MemoryLayout<Int>.size)
            let hashTableSizeTensor = graph.constant(hashTableSizeData, shape: [1], dataType: .int32)
            hashTableSizeTensors.append(hashTableSizeTensor)
            print("Level \(i): \(hashTableSize) elements")
            var weightData: Data
            if (initialHashTableWeights == nil) {
                var arr: [Float] = []
                for _ in 0..<hashTableSize {
                    arr.append(Float.random(in: -1e-4...1e-4))
                    arr.append(Float.random(in: -1e-4...1e-4))
                }
                weightData = Data(bytes: &arr, count: arr.count*MemoryLayout<Float>.size)
            } else {
                var weights = initialHashTableWeights![i]
                weightData = Data(bytes: &weights, count: weights.count*MemoryLayout<Float>.size)
            }
            
            let weightTensor = graph.variable(with: weightData, shape: [hashTableSize as NSNumber, 2 as NSNumber], dataType: .float32, name: nil)
//            let weightMomentumTensor = graph.variable(with: weightData, shape: [resolution as NSNumber, 2 as NSNumber], dataType: .float32, name: nil)
//            let weightVelocityTensor = graph.variable(with: weightData, shape: [resolution as NSNumber, 2 as NSNumber], dataType: .float32, name: nil)
            hashTableWeightTensors.append(weightTensor)
            variableTensors.append(weightTensor)
//            momentumTensors[weightTensor] = weightMomentumTensor
//            velocityTensors[weightTensor] = weightVelocityTensor
        }
        hashTableInitialized = true
    }
    
    func invokeHashTable(inputTensor: MPSGraphTensor?, dimension: Int) -> MPSGraphTensor {
        assert(hashTableInitialized)
        let prime1Data = Data(bytes: &prime1, count: MemoryLayout<Int>.size)
        let prime1Tensor = graph.constant(prime1Data, shape: [1], dataType: .int32)
        let prime2Data = Data(bytes: &prime2, count: MemoryLayout<Int>.size)
        let prime2Tensor = graph.constant(prime2Data, shape: [1], dataType: .int32)
        var interpolatedWeightTensors: [MPSGraphTensor] = []
        var inTensor: MPSGraphTensor
        if (inputTensor == nil) {
            inTensor = inputsPlaceholderTensor
        } else {
            inTensor = inputTensor!
        }
        for i in 0..<numLevels {
            let resolutionFloatTensor = graph.cast(resolutionTensors[i], to: .float32, name: nil)
            let scaledInputTensor = graph.multiplication(resolutionFloatTensor, inTensor, name: nil)
            let lowerLeftBackTensor = graph.floor(with: scaledInputTensor, name: nil)
            let lowerLeftBackIndexTensor = graph.cast(lowerLeftBackTensor, to: .int32, name: nil)
            let lowerLeftDiffTensor = graph.subtraction(scaledInputTensor, lowerLeftBackTensor, name: nil)
            let onesTensor = graph.constant(1.0, dataType: .float32)
            let oneMinusLowerLeftDiffTensor = graph.subtraction(onesTensor, lowerLeftDiffTensor, name: nil)
            
            let lowerLeftBackHashTensor = hashTensor(indexTensor: lowerLeftBackIndexTensor, primeTensor1: prime1Tensor, primeTensor2: prime2Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .ll, face: .back, dimension: dimension)
            let lowerRightBackHashTensor = hashTensor(indexTensor: lowerLeftBackIndexTensor, primeTensor1: prime1Tensor, primeTensor2: prime2Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .lr, face: .back, dimension: dimension)
            let upperLeftBackHashTensor = hashTensor(indexTensor: lowerLeftBackIndexTensor, primeTensor1: prime1Tensor, primeTensor2: prime2Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .ul, face: .back, dimension: dimension)
            let upperRightBackHashTensor = hashTensor(indexTensor: lowerLeftBackIndexTensor, primeTensor1: prime1Tensor, primeTensor2: prime2Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .ur, face: .back, dimension: dimension)
            let lowerLeftBackWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: lowerLeftBackHashTensor, batchDimensions: 0, name: nil)
            let lowerRightBackWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: lowerRightBackHashTensor, batchDimensions: 0, name: nil)
            let upperLeftBackWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: upperLeftBackHashTensor, batchDimensions: 0, name: nil)
            let upperRightBackWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: upperRightBackHashTensor, batchDimensions: 0, name: nil)
            var interpolatedWeightBack = interpolate2D(lowerLeftDiffTensor: lowerLeftDiffTensor, oneMinusLowerLeftDiffTensor: oneMinusLowerLeftDiffTensor, lowerLeftWeightTensor: lowerLeftBackWeightTensor, lowerRightWeightTensor: lowerRightBackWeightTensor, upperLeftWeightTensor: upperLeftBackWeightTensor, upperRightWeightTensor: upperRightBackWeightTensor, dimension: dimension)
            
            let lowerLeftFrontHashTensor = hashTensor(indexTensor: lowerLeftBackIndexTensor, primeTensor1: prime1Tensor, primeTensor2: prime2Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .ll, face: .front, dimension: dimension)
            let lowerRightFrontHashTensor = hashTensor(indexTensor: lowerLeftBackIndexTensor, primeTensor1: prime1Tensor, primeTensor2: prime2Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .lr, face: .front, dimension: dimension)
            let upperLeftFrontHashTensor = hashTensor(indexTensor: lowerLeftBackIndexTensor, primeTensor1: prime1Tensor, primeTensor2: prime2Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .ul, face: .front, dimension: dimension)
            let upperRightFrontHashTensor = hashTensor(indexTensor: lowerLeftBackIndexTensor, primeTensor1: prime1Tensor, primeTensor2: prime2Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .ur, face: .front, dimension: dimension)
            let lowerLeftFrontWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: lowerLeftFrontHashTensor, batchDimensions: 0, name: nil)
            let lowerRightFrontWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: lowerRightFrontHashTensor, batchDimensions: 0, name: nil)
            let upperLeftFrontWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: upperLeftFrontHashTensor, batchDimensions: 0, name: nil)
            let upperRightFrontWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: upperRightFrontHashTensor, batchDimensions: 0, name: nil)
            var interpolatedWeightFront = interpolate2D(lowerLeftDiffTensor: lowerLeftDiffTensor, oneMinusLowerLeftDiffTensor: oneMinusLowerLeftDiffTensor, lowerLeftWeightTensor: lowerLeftFrontWeightTensor, lowerRightWeightTensor: lowerRightFrontWeightTensor, upperLeftWeightTensor: upperLeftFrontWeightTensor, upperRightWeightTensor: upperRightFrontWeightTensor, dimension: dimension)
            
            let lowerLeftDiffZTensor = graph.sliceTensor(lowerLeftDiffTensor, dimension: dimension, start: 2, length: 1, name: nil)
            let oneMinusLowerLeftDiffZTensor = graph.sliceTensor(oneMinusLowerLeftDiffTensor, dimension: dimension, start: 2, length: 1, name: nil)
            interpolatedWeightBack = graph.multiplication(oneMinusLowerLeftDiffZTensor, interpolatedWeightBack, name: nil)
            interpolatedWeightFront = graph.multiplication(lowerLeftDiffZTensor, interpolatedWeightFront, name: nil)
            let interpolatedWeightTensor = graph.addition(interpolatedWeightBack, interpolatedWeightFront, name: nil)
            interpolatedWeightTensors.append(interpolatedWeightTensor)
        }
        let activationTensor = graph.concatTensors(interpolatedWeightTensors, dimension: dimension, name: nil)
        //print("Hash table activation shape: [\(activationTensor.shape![0]),\(activationTensor.shape![1])]")
        return activationTensor
    }
}

let rayThreshold: Double = -log(0.35)

class NeRFCamera : MLP3D {
    var numTrainingImages: Int
    var maxSteps: Int
    var maxT: Double
    var trainCameraParameters: Bool = false
    var deltaTPlaceholderTensor: MPSGraphTensor? = nil
    var currentCameraPos: [[Float]] = []
    var currentCameraRotation: [[Float]] = []
    var cameraPosVarTensors: [MPSGraphTensor] = []
    var cameraRotationVarTensors: [MPSGraphTensor] = []
    var imageTargetTrainingOps: [[MPSGraphOperation]] = []
    var imageTargetTrainingTensors: [[MPSGraphTensor]] = []
    var depthMapTensor: MPSGraphTensor? = nil
    var minWorldBound: [Float] = []
    var minWorldBoundTensor: MPSGraphTensor? = nil
    var worldBoundLengthTensor: MPSGraphTensor? = nil
    var worldBoundLength: Float
    init(labelSize: Int, numTrainingImages: Int, maxSteps: Int, maxT: Double, minWorldBound: [Float], worldBoundLength: Double, trainCameraParameters: Bool) {
        self.trainCameraParameters = trainCameraParameters
        self.numTrainingImages = numTrainingImages
        self.maxSteps = maxSteps
        self.maxT = maxT
        self.minWorldBound = minWorldBound
        self.worldBoundLength = Float(worldBoundLength)
        super.init(inputShape: [-1, maxSteps as NSNumber, 3], labelSize: labelSize)
        worldBoundLengthTensor = graph.constant(worldBoundLength, dataType: .float32)
        let minWorldBoundData = Data(bytes: self.minWorldBound, count: 3*MemoryLayout<Float>.size)
        minWorldBoundTensor = graph.constant(minWorldBoundData, shape: [3], dataType: .float32)
        deltaTPlaceholderTensor = graph.placeholder(shape: [maxSteps as NSNumber], dataType: .float32, name: nil)
        var initCameraPos: [Float] = [0.0, 0.0, 0.0]
        let initCameraPosData = Data(bytes: &initCameraPos, count: MemoryLayout<Float>.size*initCameraPos.count)
        var initCameraRotation: [Float] = [1.0, 0.0, 0.0,
                                           0.0, 1.0, 0.0,
                                           0.0, 0.0, 1.0]
        let initCameraRotationData = Data(bytes: &initCameraRotation, count: MemoryLayout<Float>.size*initCameraRotation.count)
        var initCameraAngle: Float = 0.0
        let initCameraAngleData = Data(bytes: &initCameraAngle, count: MemoryLayout<Float>.size)
        for _ in 0..<numTrainingImages {
            cameraPosVarTensors.append(graph.variable(with: initCameraPosData, shape: [3], dataType: .float32, name: nil))
            cameraRotationVarTensors.append(graph.variable(with: initCameraRotationData, shape: [3,3], dataType: .float32, name: nil))
            currentCameraPos.append(initCameraPos)
            currentCameraRotation.append(initCameraRotation)
        }
    }
    func loadCameraParameters(weights: CodableWeights) {
        assert(weights.cameraPos.count == weights.cameraMatrix.count)
        assert(weights.cameraPos.count == numTrainingImages)
        for i in 0..<weights.cameraPos.count {
            var cameraPos = weights.cameraPos[i]
            let cameraPosData = Data(bytes: &cameraPos, count: MemoryLayout<Float>.size*cameraPos.count)
            cameraPosVarTensors[i] = graph.variable(with: cameraPosData, shape: [3], dataType: .float32, name: nil)
            var cameraMatrix = weights.cameraMatrix[i]
            let cameraMatrixData = Data(bytes: &cameraMatrix, count: MemoryLayout<Float>.size*cameraMatrix.count)
            cameraRotationVarTensors[i] = graph.variable(with: cameraMatrixData, shape: [3,3], dataType: .float32, name: nil)
        }
    }
    override func finalize(inputTensor: MPSGraphTensor?) {
        for i in 0..<numTrainingImages {
            let ops = getImageTrainingOperations(imageIndex: i)
            imageTargetTrainingTensors.append(ops.0)
            imageTargetTrainingOps.append(ops.1)
        }
        var activationTensor: MPSGraphTensor = inputsPlaceholderTensor
        if (hashTableInitialized) {
            activationTensor = invokeHashTable(inputTensor: nil, dimension: 2)
            print("Hash table activation shape: [\(activationTensor.shape![0]),\(activationTensor.shape![1]),\(activationTensor.shape![2])]")
        } else {
            activationTensor = frequencyEncoding(inputTensor: inputsPlaceholderTensor)
        }
        //var dirModifiedTensor = graph.reshape(cameraDirPlaceholderTensor!, shape: [-1, 1, 3], name: nil)
        //dirModifiedTensor = graph.broadcast(dirModifiedTensor, shape: [-1, maxSteps as NSNumber, 3], name: nil)
        //let mlpInputTensor = graph.concatTensors([activationTensor, dirModifiedTensor], dimension: 2, name: nil)
        activationTensor = invokeMLP(inputTensor: activationTensor)
        //activationTensor = invokeMLP(inputTensor: mlpInputTensor)
        print("MLP activation shape: [\(activationTensor.shape![0]),\(activationTensor.shape![1]),\(activationTensor.shape![2])]")
        var densityTensor = graph.sliceTensor(activationTensor, dimension: 2, start: 0, length: 1, name: nil)
        var colorTensor = graph.sliceTensor(activationTensor, dimension: 2, start: 1, length: 3, name: nil)
        let deltaReshapeTensor = graph.reshape(deltaTPlaceholderTensor!, shape: [1,maxSteps as NSNumber, 1], name: nil)
        let scaledDensityTensor = graph.multiplication(densityTensor, deltaReshapeTensor, name: nil)
        print("scaledDensityTensor shape: [\(scaledDensityTensor.shape![0]),\(scaledDensityTensor.shape![1]),\(scaledDensityTensor.shape![2])]")
        var transmittanceTensor = graph.cumulativeSum(scaledDensityTensor, axis: 1, exclusive: true, reverse: false, name: nil)
        print("transmittanceTensor shape: [\(transmittanceTensor.shape![0]), \(transmittanceTensor.shape![1]), \(transmittanceTensor.shape![2])]")
        transmittanceTensor = graph.negative(with: scaledDensityTensor, name: nil)
        transmittanceTensor = graph.exponent(with: transmittanceTensor, name: nil)
        var colorScaleTensor = graph.negative(with: scaledDensityTensor, name: nil)
        colorScaleTensor = graph.exponent(with: colorScaleTensor, name: nil)
        colorScaleTensor = graph.subtraction(graph.constant(1.0, dataType: .float32), colorScaleTensor, name: nil)
        colorScaleTensor = graph.multiplication(transmittanceTensor, colorScaleTensor, name: nil)
        let unsummedColorsTensor = graph.multiplication(colorScaleTensor, colorTensor, name: nil)
        print("unsummedColorsTensor shape: [\(unsummedColorsTensor.shape![0]), \(unsummedColorsTensor.shape![1]), \(unsummedColorsTensor.shape![2])]")
        var approxColorTensor = graph.reductionSum(with: unsummedColorsTensor, axis: 1, name: nil)
        //var approxColor = graph.reductionSum(with: colorTensor, axis: 1, name: nil)
        approxColorTensor = graph.reshape(approxColorTensor, shape: [-1, 3], name: nil)
        targetInferenceTensor = approxColorTensor
        //depthMapTensor = getDepthMapTensor2(rayThreshold: rayThreshold)
        lossTensor = graph.subtraction(approxColorTensor, labelsPlaceholderTensor, name: nil)
        lossTensor = graph.multiplication(lossTensor!, lossTensor!, name: nil)
        lossTensor = graph.reductionSum(with: lossTensor!, axis: 0, name: nil)
        lossTensor = graph.reductionSum(with: lossTensor!, axis: 1, name: nil)
        targetTrainingTensors = [lossTensor!]
        let gradientTensors = graph.gradients(of: lossTensor!, with: variableTensors, name: nil)
        for variableTensor in variableTensors {
            let updateTensor = graph.stochasticGradientDescent(learningRate: graph.constant(0.001, dataType: .float32), values: variableTensor, gradient: gradientTensors[variableTensor]!, name: nil) //stochastic gradient descent for now, change to adam later
            let assignOp = graph.assign(variableTensor, tensor: updateTensor, name: nil)
            targetTrainingOps.append(assignOp)
        }
    }

    func rotateTensor(inputTensor: MPSGraphTensor, angleTensor: MPSGraphTensor, axisMatrixTensor: MPSGraphTensor) -> MPSGraphTensor {
        
        
        let sinAngleTensor = graph.sin(with: angleTensor, name: nil)
        let oneMinusCosAngleTensor = graph.subtraction(graph.constant(1.0, dataType: .float32), graph.cos(with: angleTensor, name: nil), name: nil)
        var sinSkewMatrix = graph.multiplication(sinAngleTensor, axisMatrixTensor, name: nil)
        sinSkewMatrix = graph.reshape(sinSkewMatrix, shape: [1,3,3], name: nil)
        let sinSkewMatrixMult = graph.matrixMultiplication(primary: inputTensor, secondary: sinSkewMatrix, name: nil)
        var result = graph.addition(inputTensor, sinSkewMatrixMult, name: nil)
        let cosSkewMatrix = graph.multiplication(oneMinusCosAngleTensor, axisMatrixTensor, name: nil)
        var matrixSquared = graph.matrixMultiplication(primary: cosSkewMatrix, secondary: axisMatrixTensor, name: nil)
        matrixSquared = graph.reshape(matrixSquared, shape: [1,3,3], name: nil)
        result = graph.addition(result, graph.matrixMultiplication(primary: inputTensor, secondary: matrixSquared, name: nil), name: nil)
        return result
    }
    func frequencyEncoding(inputTensor: MPSGraphTensor) -> MPSGraphTensor {
        let x = graph.sliceTensor(inputTensor, dimension: 2, start: 0, length: 1, name: nil)
        let y = graph.sliceTensor(inputTensor, dimension: 2, start: 1, length: 1, name: nil)
        let z = graph.sliceTensor(inputTensor, dimension: 2, start: 2, length: 1, name: nil)
        var tensorElements: [MPSGraphTensor] = []
        for i in 0..<5 {
            let powTensor = graph.power(graph.constant(2.0, dataType: .float32), graph.constant(Double(i), dataType: .float32), name: nil)
            let piTensor = graph.constant(Double.pi, dataType: .float32)
            var multTensor = graph.multiplication(powTensor, piTensor, name: nil)
            multTensor = graph.multiplication(multTensor, x, name: nil)
            let sinTensor = graph.sin(with: multTensor, name: nil)
            let cosTensor = graph.cos(with: multTensor, name: nil)
            tensorElements.append(sinTensor)
            tensorElements.append(cosTensor)
        }
        for i in 0..<5 {
            let powTensor = graph.power(graph.constant(2.0, dataType: .float32), graph.constant(Double(i), dataType: .float32), name: nil)
            let piTensor = graph.constant(Double.pi, dataType: .float32)
            var multTensor = graph.multiplication(powTensor, piTensor, name: nil)
            multTensor = graph.multiplication(multTensor, y, name: nil)
            let sinTensor = graph.sin(with: multTensor, name: nil)
            let cosTensor = graph.cos(with: multTensor, name: nil)
            tensorElements.append(sinTensor)
            tensorElements.append(cosTensor)
        }
        for i in 0..<6 {
            let powTensor = graph.power(graph.constant(2.0, dataType: .float32), graph.constant(Double(i), dataType: .float32), name: nil)
            let piTensor = graph.constant(Double.pi, dataType: .float32)
            var multTensor = graph.multiplication(powTensor, piTensor, name: nil)
            multTensor = graph.multiplication(multTensor, z, name: nil)
            let sinTensor = graph.sin(with: multTensor, name: nil)
            let cosTensor = graph.cos(with: multTensor, name: nil)
            tensorElements.append(sinTensor)
            tensorElements.append(cosTensor)
        }
        let resultTensor = graph.concatTensors(tensorElements, dimension: 2, name: nil)
        return resultTensor
    }
    func getImageTrainingOperations(imageIndex: Int) -> ([MPSGraphTensor], [MPSGraphOperation]) {
        assert(trainCameraParameters)
        let cameraPosTensor = cameraPosVarTensors[imageIndex]
        //let inputTensorRotated = rotateTensor(inputTensor: inputsPlaceholderTensor, angleTensor: angleTensor, axisMatrixTensor: axisMatrixTensor)
        let inputTensorRotated = graph.matrixMultiplication(primary: inputsPlaceholderTensor, secondary: cameraRotationVarTensors[imageIndex], name: nil)
        var modifiedInputTensor = graph.addition(cameraPosTensor, inputTensorRotated, name: nil)
        modifiedInputTensor = graph.subtraction(modifiedInputTensor, minWorldBoundTensor!, name: nil)
        modifiedInputTensor = graph.division(modifiedInputTensor, worldBoundLengthTensor!, name: nil)
        //var dirModifiedTensor = graph.reshape(cameraDirPlaceholderTensor!, shape: [-1, 1, 3], name: nil)
        //dirModifiedTensor = graph.broadcast(dirModifiedTensor, shape: [-1, maxSteps as NSNumber, 3], name: nil)
        //let mlpInputTensor = graph.concatTensors([activationTensor, dirModifiedTensor], dimension: 2, name: nil)
        //var activationTensor: MPSGraphTensor = modifiedInputTensor
        var activationTensor: MPSGraphTensor = frequencyEncoding(inputTensor: modifiedInputTensor)
//        if (hashTableInitialized) {
//            activationTensor = invokeHashTable(inputTensor: modifiedInputTensor, dimension: 2)
//            print("Hash table activation shape: [\(activationTensor.shape![0]),\(activationTensor.shape![1]),\(activationTensor.shape![2])]")
//        }
        activationTensor = invokeMLP(inputTensor: activationTensor)
        //activationTensor = invokeMLP(inputTensor: mlpInputTensor)
        //print("MLP activation shape: [\(activationTensor.shape![0]),\(activationTensor.shape![1]),\(activationTensor.shape![2])]")
        var densityTensor = graph.sliceTensor(activationTensor, dimension: 2, start: 0, length: 1, name: nil)
        var colorTensor = graph.sliceTensor(activationTensor, dimension: 2, start: 1, length: 3, name: nil)
        let deltaReshapeTensor = graph.reshape(deltaTPlaceholderTensor!, shape: [1,maxSteps as NSNumber, 1], name: nil)
        let scaledDensityTensor = graph.multiplication(densityTensor, deltaReshapeTensor, name: nil)
        //print("scaledDensityTensor shape: [\(scaledDensityTensor.shape![0]),\(scaledDensityTensor.shape![1]),\(scaledDensityTensor.shape![2])]")
        var transmittanceTensor = graph.cumulativeSum(scaledDensityTensor, axis: 1, exclusive: true, reverse: false, name: nil)
        //print("transmittanceTensor shape: [\(transmittanceTensor.shape![0]), \(transmittanceTensor.shape![1]), \(transmittanceTensor.shape![2])]")
        transmittanceTensor = graph.negative(with: scaledDensityTensor, name: nil)
        transmittanceTensor = graph.exponent(with: transmittanceTensor, name: nil)
        var colorScaleTensor = graph.negative(with: scaledDensityTensor, name: nil)
        colorScaleTensor = graph.exponent(with: colorScaleTensor, name: nil)
        colorScaleTensor = graph.subtraction(graph.constant(1.0, dataType: .float32), colorScaleTensor, name: nil)
        colorScaleTensor = graph.multiplication(transmittanceTensor, colorScaleTensor, name: nil)
        let unsummedColorsTensor = graph.multiplication(colorScaleTensor, colorTensor, name: nil)
        //print("unsummedColorsTensor shape: [\(unsummedColorsTensor.shape![0]), \(unsummedColorsTensor.shape![1]), \(unsummedColorsTensor.shape![2])]")
        var approxColorTensor = graph.reductionSum(with: unsummedColorsTensor, axis: 1, name: nil)
        //var approxColor = graph.reductionSum(with: colorTensor, axis: 1, name: nil)
        approxColorTensor = graph.reshape(approxColorTensor, shape: [-1, 3], name: nil)
        targetInferenceTensor = approxColorTensor
        var loss = graph.subtraction(approxColorTensor, labelsPlaceholderTensor, name: nil)
        loss = graph.multiplication(loss, loss, name: nil)
        loss = graph.reductionSum(with: loss, axis: 0, name: nil)
        loss = graph.reductionSum(with: loss, axis: 1, name: nil)
//        let squaredMatrix = graph.multiplication(cameraRotationVarTensors[imageIndex], cameraRotationVarTensors[imageIndex], name: nil)
//        var matrixReduction = graph.reductionSum(with: squaredMatrix, axis: 0, name: nil)
//        matrixReduction = graph.reductionSum(with: matrixReduction, axis: 1, name: nil)
//        matrixReduction = graph.squeeze(matrixReduction, name: nil)
//        loss = graph.addition(loss, matrixReduction, name: nil)
        if (imageIndex != 0) {
            let cameraMatrixTranspose = graph.transposeTensor(cameraRotationVarTensors[imageIndex], dimension: 0, withDimension: 1, name: nil)
            let cameraMatrixMult = graph.matrixMultiplication(primary: cameraRotationVarTensors[imageIndex], secondary: cameraMatrixTranspose, name: nil)
            var identityMatrix: [Float] = [1.0, 0.0, 0.0,
                                           0.0, 1.0, 0.0,
                                           0.0, 0.0, 1.0]
            let identityMatrixData = Data(bytes: &identityMatrix, count: identityMatrix.count*MemoryLayout<Float>.size)
            let identityMatrixTensor = graph.constant(identityMatrixData, shape: [3,3], dataType: .float32)
            var matrixSub = graph.subtraction(cameraMatrixMult, identityMatrixTensor, name: nil)
            //matrixSub = graph.multiplication(matrixSub, matrixSub, name: nil)
            matrixSub = graph.absolute(with: matrixSub, name: nil)
            var matrixReduction = graph.reductionSum(with: matrixSub, axis: 0, name: nil)
            matrixReduction = graph.reductionSum(with: matrixReduction, axis: 1, name: nil)
            matrixReduction = graph.squeeze(matrixReduction, name: nil)
            matrixReduction = graph.multiplication(graph.constant(0.001, dataType: .float32), matrixReduction, name: nil)
            loss = graph.addition(loss, matrixReduction, name: nil)
        }
        let targets: [MPSGraphTensor] = [loss, cameraPosVarTensors[imageIndex],
                                         cameraRotationVarTensors[imageIndex]]
        var trainingOps: [MPSGraphOperation] = []
        var gradientInputTensors: [MPSGraphTensor] = []
        if (imageIndex != 0) {
            gradientInputTensors.append(cameraRotationVarTensors[imageIndex])
            gradientInputTensors.append(cameraPosVarTensors[imageIndex])
        }
        gradientInputTensors.append(contentsOf: layerWeightTensors)
        gradientInputTensors.append(contentsOf: layerBiasTensors)
        let gradientTensors = graph.gradients(of: loss, with: gradientInputTensors, name: nil)
        for variableTensor in gradientInputTensors {
            let updateTensor = graph.stochasticGradientDescent(learningRate: graph.constant(0.001, dataType: .float32), values: variableTensor, gradient: gradientTensors[variableTensor]!, name: nil) //stochastic gradient descent for now, change to adam later
            let assignOp = graph.assign(variableTensor, tensor: updateTensor, name: nil)
            trainingOps.append(assignOp)
        }
        return (targets, trainingOps)
    }
    func trainCameraParameterDoubleBufferedDynamic(imageIndex: Int, imageWidth: Int, imageHeight: Int, inputsLabels: [([Float], [Float])], numEpochs: Int, batchSize: Int, focalLength: Float) {
        //assert(!hashTableInitialized)
        assert(inputsLabels.count == imageWidth*imageHeight)
        var shuffledInputsLabels = inputsLabels
        shuffledInputsLabels.shuffle()
        let mpsDevice = MPSGraphDevice(mtlDevice: device!)
        let execDescriptor = MPSGraphExecutionDescriptor()
        execDescriptor.completionHandler = { (resultsDictionary, nil) in
            self.doubleBufferingSemaphore.signal()
            let resultLoss = resultsDictionary[self.imageTargetTrainingTensors[imageIndex][0]]
            var loss: [Float] = Array(repeating: 0.0, count: 1)
            resultLoss!.mpsndarray().readBytes(&loss, strideBytes: nil)
            self.loss += loss[0]
            var resultCameraPos: [Float] = Array(repeating: 0.0, count: 3)
            var resultCameraMatrix: [Float] = Array(repeating: 0.0, count: 9)
            resultsDictionary[self.cameraPosVarTensors[imageIndex]]!.mpsndarray().readBytes(&resultCameraPos, strideBytes: nil)
            resultsDictionary[self.cameraRotationVarTensors[imageIndex]]!.mpsndarray().readBytes(&resultCameraMatrix, strideBytes: nil)
            self.currentCameraPos[imageIndex] = resultCameraPos
            self.currentCameraRotation[imageIndex] = resultCameraMatrix
        }
        var deltaValues: [Float] = []
        var currentT: Float = 1.0
        for i in 0..<maxSteps {
            let b = exp( (log(maxT) - log(1.0)) / Double(maxSteps))
            let newT = Float(1.0*pow(b, Double(i+1)))
            let deltaT = newT-currentT
            deltaValues.append(deltaT)
            currentT = newT
            print("t[\(i)]: \(currentT), deltaT[\(i)]: \(deltaT)")
        }
        let deltaValuesData = Data(bytes: &deltaValues, count: deltaValues.count*MemoryLayout<Float>.size)
        let deltaValuesTensorData = MPSGraphTensorData(device: mpsDevice, data: deltaValuesData, shape: [maxSteps as NSNumber], dataType: .float32)
        
        for i in 1...numEpochs {
            loss = 0.0
            var firstIndex = 0
            while (firstIndex < shuffledInputsLabels.count) {
                autoreleasepool {
                    var inputBatch: [Float] = []
                    var labelBatch: [Float] = []
                    let curBatchSize: Int = min(batchSize, shuffledInputsLabels.count-firstIndex)

                    for pair in inputsLabels[firstIndex..<firstIndex+curBatchSize] {
                        let rayDir = simd_float3(x: pair.0[0] - 0.5, y: pair.0[1] - 0.5, z: -focalLength)
                        let raySteps = castRay(cameraDir: rayDir, cameraPos: simd_float3(x: 0.0, y: 0.0, z: 0.0), numSteps: maxSteps, minWorldBound: simd_float3(x: minWorldBound[0], y: minWorldBound[1], z: minWorldBound[2]), maxWorldBound: simd_float3(x: minWorldBound[0] + worldBoundLength, y: minWorldBound[1] + worldBoundLength, z: minWorldBound[2] + worldBoundLength), deltaValues: deltaValues)
                        inputBatch.append(contentsOf: raySteps)
                    }
                    
                    for pair in inputsLabels[firstIndex..<firstIndex+curBatchSize] {
                        for elem in pair.1 {
                            labelBatch.append(elem)
                        }
                    }
                    
                    doubleBufferingSemaphore.wait()
                    let inputTensorDataTemp = Data(bytes: &inputBatch, count: MemoryLayout<Float>.size*inputBatch.count)
                    let inputTensorDataCurrent = MPSGraphTensorData(device: mpsDevice, data: inputTensorDataTemp, shape: [curBatchSize as NSNumber, maxSteps as NSNumber, 3], dataType: .float32)
                    //print(inputTensorData[currentTensorDataIndex].shape)
                    let labelTensorDataTemp = Data(bytes: &labelBatch, count: MemoryLayout<Float>.size*labelBatch.count)
                    let labelTensorDataCurrent = MPSGraphTensorData(device: mpsDevice, data: labelTensorDataTemp, shape: [curBatchSize as NSNumber, inputsLabels[0].1.count as NSNumber], dataType: .float32)
                    graph.runAsync(with: commandQueue!, feeds: [inputsPlaceholderTensor: inputTensorDataCurrent, labelsPlaceholderTensor: labelTensorDataCurrent, deltaTPlaceholderTensor!: deltaValuesTensorData], targetTensors: imageTargetTrainingTensors[imageIndex], targetOperations: imageTargetTrainingOps[imageIndex], executionDescriptor: execDescriptor)
                }
                firstIndex += batchSize
            }
            print("Loss: \(loss)")
            shuffledInputsLabels.shuffle()
        }
    }
    func trainDoubleBufferedDynamic(imageWidth: Int, imageHeight: Int, inputData: [[Float]], labelData: [[Float]], numLabels: Int, numEpochs: Int, batchSize: Int, cameraPos: simd_float3, cameraAxis: simd_float3, cameraAngle: Float, focalLength: Float, maxT: Double, minWorldBound: simd_float3, maxWorldBound: simd_float3) {
        assert(inputData.count == labelData.count)
        assert(inputData.count == numLabels)
        assert(inputData.count == imageWidth*imageHeight)
        assert(inputData[0].count == 2)
        assert(labelData[0].count == labelSize)
        var shuffledIndices = inputData.indices.shuffled()
        var inputDataShuffled = shuffledIndices.map{inputData[$0]}
        var labelDataShuffled = shuffledIndices.map{labelData[$0]}
        let mpsDevice = MPSGraphDevice(mtlDevice: device!)
        let execDescriptor = MPSGraphExecutionDescriptor()
        execDescriptor.completionHandler = { (resultsDictionary, nil) in
            self.doubleBufferingSemaphore.signal()
            if (self.lossTensor != nil) {
                let result = resultsDictionary[self.lossTensor!]
                var loss: [Float] = Array(repeating: 0.0, count: 1)
                result!.mpsndarray().readBytes(&loss, strideBytes: nil)
                self.loss += loss[0]
            }
        }
        var deltaValues: [Float] = []
        var currentT: Float = 1.0
        for i in 0..<maxSteps {
            let b = exp( (log(maxT) - log(1.0)) / Double(maxSteps))
            let newT = Float(1.0*pow(b, Double(i+1)))
            let deltaT = newT-currentT
            deltaValues.append(deltaT)
            currentT = newT
            print("t[\(i)]: \(currentT), deltaT[\(i)]: \(deltaT)")
        }
        let deltaValuesData = Data(bytes: &deltaValues, count: deltaValues.count*MemoryLayout<Float>.size)
        let deltaValuesTensorData = MPSGraphTensorData(device: mpsDevice, data: deltaValuesData, shape: [maxSteps as NSNumber], dataType: .float32)
        for _ in 1...numEpochs {
            loss = 0.0
            var firstIndex = 0
            while (firstIndex < numLabels) {
                autoreleasepool {
                    var inputBatch: [Float] = []
                    var labelBatch: [Float] = []
                    let curBatchSize: Int = min(batchSize, numLabels-firstIndex)

                    for arr in inputDataShuffled[firstIndex..<firstIndex+curBatchSize] {
                        let rayDir = simd_float3(x: arr[0] - 0.5, y: arr[1] - 0.5, z: -focalLength)
                        let rayDirRot = rotateVector(dir: rayDir, axis: cameraAxis, angle: cameraAngle)
                        let raySteps = castRay(cameraDir: rayDirRot, cameraPos: cameraPos, numSteps: maxSteps, minWorldBound: minWorldBound, maxWorldBound: maxWorldBound, deltaValues: deltaValues)
                        inputBatch.append(contentsOf: raySteps)
                    }
                    
                    for arr in labelDataShuffled[firstIndex..<firstIndex+curBatchSize] {
                        for elem in arr {
                            labelBatch.append(elem)
                        }
                    }
                    
                    doubleBufferingSemaphore.wait()
                    let inputTensorDataTemp = Data(bytes: &inputBatch, count: MemoryLayout<Float>.size*inputBatch.count)
                    let inputTensorDataCurrent = MPSGraphTensorData(device: mpsDevice, data: inputTensorDataTemp, shape: [curBatchSize as NSNumber, maxSteps as NSNumber, 3], dataType: .float32)
                    //print(inputTensorData[currentTensorDataIndex].shape)
                    let labelTensorDataTemp = Data(bytes: &labelBatch, count: MemoryLayout<Float>.size*labelBatch.count)
                    let labelTensorDataCurrent = MPSGraphTensorData(device: mpsDevice, data: labelTensorDataTemp, shape: [curBatchSize as NSNumber, labelData[0].count as NSNumber], dataType: .float32)
                    graph.runAsync(with: commandQueue!, feeds: [inputsPlaceholderTensor: inputTensorDataCurrent, labelsPlaceholderTensor: labelTensorDataCurrent, deltaTPlaceholderTensor!: deltaValuesTensorData], targetTensors: targetTrainingTensors, targetOperations: targetTrainingOps, executionDescriptor: execDescriptor)
                }
                firstIndex += batchSize
            }
            shuffledIndices.shuffle()
            inputDataShuffled = shuffledIndices.map{inputData[$0]}
            labelDataShuffled = shuffledIndices.map{labelData[$0]}
            print("Loss: \(loss)")
        }
        
    }
    
    func getInferenceDepthMapPiecemealDynamic(imageWidth: Int, imageHeight: Int, inputData: [[Float]], batchSize: Int, cameraPos: simd_float3, cameraAxis: simd_float3, cameraAngle: Float, focalLength: Float, maxT: Double, minWorldBound: simd_float3, maxWorldBound: simd_float3, getWeights: Bool, resultsOut: inout [MPSGraphTensor: MPSGraphTensorData]) -> ([Float]) {
        assert(inputData.count == imageWidth*imageHeight)
        assert(inputData[0].count == 2)
        let mpsDevice = MPSGraphDevice(mtlDevice: device!)
        var results: [Float] = []
        //var depthResults: [Float] = []
        var deltaValues: [Float] = []
        var currentT: Float = 1.0
        for i in 0..<maxSteps {
            let b = exp( (log(maxT) - log(1.0)) / Double(maxSteps))
            let newT = Float(1.0*pow(b, Double(i+1)))
            let deltaT = newT-currentT
            deltaValues.append(deltaT)
            currentT = newT
            //print("t[\(i)]: \(currentT), deltaT[\(i)]: \(deltaT)")
        }
        let deltaValuesData = Data(bytes: &deltaValues, count: deltaValues.count*MemoryLayout<Float>.size)
        let deltaValuesTensorData = MPSGraphTensorData(device: mpsDevice, data: deltaValuesData, shape: [maxSteps as NSNumber], dataType: .float32)
        var firstIndex = 0
        while (firstIndex < inputData.count) {
            autoreleasepool {
                var inputBatch: [Float] = []
                var dirBatch: [Float] = []
                let curBatchSize: Int = min(batchSize, inputData.count-firstIndex)

                for arr in inputData[firstIndex..<firstIndex+curBatchSize] {
                    let rayDir = simd_float3(x: arr[0] - 0.5, y: arr[1] - 0.5, z: -focalLength)
                    let rayDirRot = rotateVector(dir: rayDir, axis: cameraAxis, angle: cameraAngle)
                    let raySteps = castRay(cameraDir: rayDirRot, cameraPos: cameraPos, numSteps: maxSteps, minWorldBound: minWorldBound, maxWorldBound: maxWorldBound, deltaValues: deltaValues)
                    inputBatch.append(contentsOf: raySteps)
                    let rayDirNormalized = simd_normalize(rayDir)
                    dirBatch.append(rayDirNormalized.x)
                    dirBatch.append(rayDirNormalized.y)
                    dirBatch.append(rayDirNormalized.z)
                }
                let inputTensorDataTemp = Data(bytes: &inputBatch, count: MemoryLayout<Float>.size*inputBatch.count)
                let inputTensorDataCurrent = MPSGraphTensorData(device: mpsDevice, data: inputTensorDataTemp, shape: [curBatchSize as NSNumber, maxSteps as NSNumber, 3], dataType: .float32)
                let dirTensorDataTemp = Data(bytes: &dirBatch, count: MemoryLayout<Float>.size*dirBatch.count)
                let dirTensorDataCurrent = MPSGraphTensorData(device: mpsDevice, data: dirTensorDataTemp, shape: [curBatchSize as NSNumber, 3], dataType: .float32)
                var targetTensors: [MPSGraphTensor] = [targetInferenceTensor!]
                if (getWeights) {
                    targetTensors.append(contentsOf: layerWeightTensors)
                    targetTensors.append(contentsOf: layerBiasTensors)
                    targetTensors.append(contentsOf: hashTableWeightTensors)
                }
                let resultDict = graph.run(with: commandQueue!, feeds: [inputsPlaceholderTensor: inputTensorDataCurrent, deltaTPlaceholderTensor!: deltaValuesTensorData], targetTensors: targetTensors, targetOperations: nil)
                //let depthResultsDict = graph.run(with: commandQueue!, feeds: [inputsPlaceholderTensor: inputTensorDataCurrent, cameraDirPlaceholderTensor!: dirTensorDataCurrent, deltaTPlaceholderTensor!: deltaValuesTensorData], targetTensors: [depthMapTensor!], targetOperations: nil)
                resultsOut = resultDict
                var resultArr: [Float] = Array(repeating: 0.0, count: curBatchSize*self.labelSize)
                resultDict[targetInferenceTensor!]!.mpsndarray().readBytes(&resultArr, strideBytes: nil)
                //var depthResultArr: [Float] = Array(repeating: 0.0, count: curBatchSize)
                //depthResultsDict[depthMapTensor!]!.mpsndarray().readBytes(&depthResultArr, strideBytes: nil)
                results.append(contentsOf: resultArr)
                //depthResults.append(contentsOf: depthResultArr)
            }
            firstIndex += batchSize
        }
        //return (results, depthResults)
        return results
    }
    override func getVariableData(results: [MPSGraphTensor: MPSGraphTensorData]) -> Data {
        var weights = CodableWeights()
        for i in 0..<layerWeightTensors.count {
            var weightMatrix: [Float] = Array(repeating: 0.0, count: layers[i].numInputs*layers[i].numOutputs)
            let tensorData: MPSGraphTensorData = results[layerWeightTensors[i]]!
            tensorData.mpsndarray().readBytes(&weightMatrix, strideBytes: nil)
            weights.weightMatrices.append(weightMatrix)
        }
        for i in 0..<layerBiasTensors.count {
            var biasVector: [Float] = Array(repeating: 0.0, count: layers[i].numOutputs)
            let tensorData: MPSGraphTensorData = results[layerBiasTensors[i]]!
            tensorData.mpsndarray().readBytes(&biasVector, strideBytes: nil)
            weights.biasVectors.append(biasVector)
        }
        for i in 0..<hashTableWeightTensors.count {
            var hashTableWeights: [Float] = Array(repeating: 0.0, count: hashTableWeightTensors[i].shape![0].intValue*hashTableWeightTensors[i].shape![1].intValue)
            let tensorData: MPSGraphTensorData = results[hashTableWeightTensors[i]]!
            tensorData.mpsndarray().readBytes(&hashTableWeights, strideBytes: nil)
            weights.hashTableWeights.append(hashTableWeights)
        }
        for i in 0..<numTrainingImages {
            weights.cameraMatrix.append(currentCameraRotation[i])
            weights.cameraPos.append(currentCameraPos[i])
        }
        let jsonEncoder = JSONEncoder()
        let jsonData = try! jsonEncoder.encode(weights)
        return jsonData
    }
}

class NeRF : MLP3D {
    var numTrainingImages: Int
    var deltaTPlaceholderTensor: MPSGraphTensor? = nil
    var cameraDirPlaceholderTensor: MPSGraphTensor? = nil
    var maxSteps: Int
    var maxT: Double
    var depthMapTensor: MPSGraphTensor? = nil
    init(labelSize: Int, numTrainingImages: Int, maxSteps: Int, maxT: Double) {
        self.numTrainingImages = numTrainingImages
        self.maxSteps = maxSteps
        self.maxT = maxT
        super.init(inputShape: [-1, maxSteps as NSNumber, 3], labelSize: labelSize)
    }
    override func finalize(inputTensor: MPSGraphTensor?) {
        deltaTPlaceholderTensor = graph.placeholder(shape: [maxSteps as NSNumber], dataType: .float32, name: nil)
        cameraDirPlaceholderTensor = graph.placeholder(shape: [-1,3], dataType: .float32, name: nil)
        var activationTensor: MPSGraphTensor = inputsPlaceholderTensor
        if (hashTableInitialized) {
            activationTensor = invokeHashTable(inputTensor: nil, dimension: 2)
            print("Hash table activation shape: [\(activationTensor.shape![0]),\(activationTensor.shape![1]),\(activationTensor.shape![2])]")
        }
        //var dirModifiedTensor = graph.reshape(cameraDirPlaceholderTensor!, shape: [-1, 1, 3], name: nil)
        //dirModifiedTensor = graph.broadcast(dirModifiedTensor, shape: [-1, maxSteps as NSNumber, 3], name: nil)
        //let mlpInputTensor = graph.concatTensors([activationTensor, dirModifiedTensor], dimension: 2, name: nil)
        activationTensor = invokeMLP(inputTensor: activationTensor)
        //activationTensor = invokeMLP(inputTensor: mlpInputTensor)
        print("MLP activation shape: [\(activationTensor.shape![0]),\(activationTensor.shape![1]),\(activationTensor.shape![2])]")
        var densityTensor = graph.sliceTensor(activationTensor, dimension: 2, start: 0, length: 1, name: nil)
        var colorTensor = graph.sliceTensor(activationTensor, dimension: 2, start: 1, length: 3, name: nil)
        let deltaReshapeTensor = graph.reshape(deltaTPlaceholderTensor!, shape: [1,maxSteps as NSNumber, 1], name: nil)
        let scaledDensityTensor = graph.multiplication(densityTensor, deltaReshapeTensor, name: nil)
        print("scaledDensityTensor shape: [\(scaledDensityTensor.shape![0]),\(scaledDensityTensor.shape![1]),\(scaledDensityTensor.shape![2])]")
        var transmittanceTensor = graph.cumulativeSum(scaledDensityTensor, axis: 1, exclusive: true, reverse: false, name: nil)
        print("transmittanceTensor shape: [\(transmittanceTensor.shape![0]), \(transmittanceTensor.shape![1]), \(transmittanceTensor.shape![2])]")
        transmittanceTensor = graph.negative(with: scaledDensityTensor, name: nil)
        transmittanceTensor = graph.exponent(with: transmittanceTensor, name: nil)
        var colorScaleTensor = graph.negative(with: scaledDensityTensor, name: nil)
        colorScaleTensor = graph.exponent(with: colorScaleTensor, name: nil)
        colorScaleTensor = graph.subtraction(graph.constant(1.0, dataType: .float32), colorScaleTensor, name: nil)
        colorScaleTensor = graph.multiplication(transmittanceTensor, colorScaleTensor, name: nil)
        let unsummedColorsTensor = graph.multiplication(colorScaleTensor, colorTensor, name: nil)
        print("unsummedColorsTensor shape: [\(unsummedColorsTensor.shape![0]), \(unsummedColorsTensor.shape![1]), \(unsummedColorsTensor.shape![2])]")
        var approxColorTensor = graph.reductionSum(with: unsummedColorsTensor, axis: 1, name: nil)
        //var approxColor = graph.reductionSum(with: colorTensor, axis: 1, name: nil)
        approxColorTensor = graph.reshape(approxColorTensor, shape: [-1, 3], name: nil)
        targetInferenceTensor = approxColorTensor
        depthMapTensor = getDepthMapTensor2(rayThreshold: rayThreshold)
        lossTensor = graph.subtraction(approxColorTensor, labelsPlaceholderTensor, name: nil)
        lossTensor = graph.multiplication(lossTensor!, lossTensor!, name: nil)
        lossTensor = graph.reductionSum(with: lossTensor!, axis: 0, name: nil)
        lossTensor = graph.reductionSum(with: lossTensor!, axis: 1, name: nil)
        targetTrainingTensors = [lossTensor!]
        let gradientTensors = graph.gradients(of: lossTensor!, with: variableTensors, name: nil)
        for variableTensor in variableTensors {
            let updateTensor = graph.stochasticGradientDescent(learningRate: graph.constant(0.001, dataType: .float32), values: variableTensor, gradient: gradientTensors[variableTensor]!, name: nil) //stochastic gradient descent for now, change to adam later
            let assignOp = graph.assign(variableTensor, tensor: updateTensor, name: nil)
            targetTrainingOps.append(assignOp)
        }
    }
    func getDepthMapTensor(deltaStepScale: Double, rayThreshold: Double, maxT: Double, depthInputTensor: MPSGraphTensor, cameraPosTensor: MPSGraphTensor) -> MPSGraphTensor {
        let transmittanceThresholdTensor = graph.constant(rayThreshold, dataType: .float32)
        let maxTTensor = graph.constant(maxT, dataType: .float32)
        var maxStepsTensor = graph.constant(Double(maxSteps), dataType: .int32)
        let whileBeforeBlock: MPSGraphWhileBeforeBlock = { [self] (inputTensors: [MPSGraphTensor], resultsTensor: NSMutableArray) -> MPSGraphTensor in
            let rayPos = inputTensors[0]
            let accumulatedTransmittanceTensor = inputTensors[1]
            let iterationTensor = inputTensors[2]
            let tTensor = inputTensors[3]
            let rayDir = inputTensors[4]
            resultsTensor.removeAllObjects()
            resultsTensor.add(rayPos)
            resultsTensor.add(accumulatedTransmittanceTensor)
            resultsTensor.add(iterationTensor)
            resultsTensor.add(tTensor)
            resultsTensor.add(rayDir)
            let minAccumulatedTransmittanceTensor = self.graph.reductionMinimum(with: accumulatedTransmittanceTensor, axis: 0, name: nil)
            let bool1Tensor = self.graph.lessThan(minAccumulatedTransmittanceTensor, transmittanceThresholdTensor, name: nil)
            let bool2Tensor = self.graph.lessThan(iterationTensor, maxStepsTensor, name: nil)
            return self.graph.logicalAND(bool1Tensor, bool2Tensor, name: nil)
        }
        let whileAfterBlock: MPSGraphWhileAfterBlock = { [self] (bodyBlockArguments: [MPSGraphTensor]) -> [MPSGraphTensor] in
            var results: [MPSGraphTensor] = []
            let rayPos = bodyBlockArguments[0]
            var accumulatedTransmittanceTensor = bodyBlockArguments[1]
            var iterationTensor = bodyBlockArguments[2]
            var tTensor = bodyBlockArguments[3]
            let rayDir = bodyBlockArguments[4]
            let deltaScaleTensor = graph.constant(deltaStepScale, dataType: .float32)
            let deltaTTensor = graph.multiplication(deltaScaleTensor, tTensor, name: nil)
            let posTensor = self.graph.addition(rayPos, graph.multiplication(deltaTTensor, rayDir, name: nil), name: nil)
            let hashEncoding = self.invokeHashTable(inputTensor: posTensor, dimension: 1)
            let activationTensor = self.invokeMLP(inputTensor: hashEncoding)
            var densityTensor = graph.sliceTensor(activationTensor, dimension: 1, start: 0, length: 1, name: nil)
            var deltaTransmittanceTensor = graph.multiplication(deltaTTensor, densityTensor, name: nil)
            let boolTensor = graph.lessThan(accumulatedTransmittanceTensor, transmittanceThresholdTensor, name: nil)
            accumulatedTransmittanceTensor = graph.addition(accumulatedTransmittanceTensor, deltaTransmittanceTensor, name: nil)
            tTensor = self.graph.addition(tTensor, graph.multiplication(boolTensor, deltaTTensor, name: nil), name: nil)
            var oneTensor = self.graph.constant(1, dataType: .float32)
            oneTensor = self.graph.cast(oneTensor, to: .int32, name: nil)
            iterationTensor = self.graph.addition(iterationTensor, oneTensor, name: nil)
            results.append(rayPos)
            results.append(accumulatedTransmittanceTensor)
            results.append(iterationTensor)
            results.append(tTensor)
            results.append(rayDir)
            return results
        }
        var initialInputs: [MPSGraphTensor] = []
        initialInputs.append(cameraPosTensor)
        var initialT: Float = 1.0
        let initialTData = Data(bytes: &initialT, count: MemoryLayout<Float>.size)
        var tTensor = graph.constant(initialTData, shape: [1], dataType: .float32)
        tTensor = graph.broadcast(tTensor, shape: [depthInputTensor.shape![0], 1], name: nil)
        var initialTransmittance: Float = 0.0
        let initialTransmittanceData = Data(bytes: &initialTransmittance, count: MemoryLayout<Float>.size)
        var initialAccumulatedTransmittanceTensor = graph.constant(initialTransmittanceData, shape: [1], dataType: .float32)
        initialAccumulatedTransmittanceTensor = graph.broadcast(initialAccumulatedTransmittanceTensor, shape: [depthInputTensor.shape![0],1], name: nil)
        initialInputs.append(initialAccumulatedTransmittanceTensor)
        var initialI: Int = 0
        let initialIData = Data(bytes: &initialI, count: MemoryLayout<Int>.size)
        let iterationTensor = graph.constant(initialIData, shape: [1], dataType: .int32)
        initialInputs.append(iterationTensor)
        initialInputs.append(tTensor)
        initialInputs.append(depthInputTensor)
        let resultsTensors: [MPSGraphTensor] = graph.while(initialInputs: initialInputs, before: whileBeforeBlock, after: whileAfterBlock, name: nil)
        //let resultsTensors = initialInputs
        let accumulatedTTensor = resultsTensors[3]
        print("accumulatedTTensor shape: [\(accumulatedTTensor.shape![0]), \(accumulatedTTensor.shape![1])]")
        return graph.division(accumulatedTTensor, maxTTensor, name: nil)
    }
    func getDepthMapTensor2(rayThreshold: Double) -> MPSGraphTensor {
        let transmittanceThresholdTensor = graph.constant(rayThreshold, dataType: .float32)
        var activationTensor: MPSGraphTensor = inputsPlaceholderTensor
        if (hashTableInitialized) {
            activationTensor = invokeHashTable(inputTensor: nil, dimension: 2)
            print("Hash table activation shape: [\(activationTensor.shape![0]),\(activationTensor.shape![1]),\(activationTensor.shape![2])]")
        }
        //var dirModifiedTensor = graph.reshape(cameraDirPlaceholderTensor!, shape: [-1, 1, 3], name: nil)
        //dirModifiedTensor = graph.broadcast(dirModifiedTensor, shape: [-1, maxSteps as NSNumber, 3], name: nil)
        //let mlpInputTensor = graph.concatTensors([activationTensor, dirModifiedTensor], dimension: 2, name: nil)
        activationTensor = invokeMLP(inputTensor: activationTensor)
        //activationTensor = invokeMLP(inputTensor: mlpInputTensor)
        print("MLP activation shape: [\(activationTensor.shape![0]),\(activationTensor.shape![1]),\(activationTensor.shape![2])]")
        let densityTensor = graph.sliceTensor(activationTensor, dimension: 2, start: 0, length: 1, name: nil)
        let deltaReshapeTensor = graph.reshape(deltaTPlaceholderTensor!, shape: [1,maxSteps as NSNumber, 1], name: nil)
        let scaledDensityTensor = graph.multiplication(densityTensor, deltaReshapeTensor, name: nil)
        print("scaledDensityTensor shape: [\(scaledDensityTensor.shape![0]),\(scaledDensityTensor.shape![1]),\(scaledDensityTensor.shape![2])]")
        let transmittanceTensor = graph.cumulativeSum(scaledDensityTensor, axis: 1, exclusive: true, reverse: false, name: nil)
        let booleanTensor = graph.lessThan(transmittanceTensor, transmittanceThresholdTensor, name: nil)
        let subTensor = graph.subtraction(transmittanceTensor, transmittanceThresholdTensor, name: nil)
        var signTensor = graph.sign(with: subTensor, name: nil)
        //let selectTensor = graph.select(predicate: booleanTensor, trueTensor: graph.constant(0.0, dataType: .float32), falseTensor: graph.constant(1.0, dataType: .float32), name: nil)
        signTensor = graph.addition(signTensor, graph.constant(1.0, dataType: .float32), name: nil)
        signTensor = graph.division(signTensor, graph.constant(2.0, dataType: .float32), name: nil)
        let sumTensor = graph.reductionSum(with: signTensor, axis: 1, name: nil)
//        var sumTensor = graph.reductionSum(with: scaledDensityTensor, axis: 1, name: nil)
//        sumTensor = graph.negative(with: sumTensor, name: nil)
//        sumTensor = graph.exponent(with: sumTensor, name: nil)
        let resultTensor = graph.reshape(sumTensor, shape: [-1,1], name: nil)
        return resultTensor
    }
    func trainDoubleBuffered(inputData: [[Float]], labelData: [[Float]], dirData: [[Float]], deltaValues: inout [Float], numLabels: Int, numEpochs: Int, batchSize: Int, inputDataType: MPSDataType) {
        assert(inputData.count == labelData.count)
        assert(labelData.count == dirData.count)
        assert(inputData[0].count == 3*maxSteps)
        assert(labelData[0].count == labelSize)
        assert(dirData[0].count == 3)
        var shuffledIndices = inputData.indices.shuffled()
        var inputDataShuffled = shuffledIndices.map{inputData[$0]}
        var labelDataShuffled = shuffledIndices.map{labelData[$0]}
        var dirDataShuffled = shuffledIndices.map{dirData[$0]}
        var inputBatch: [Float] = []
        var labelBatch: [Float] = []
        var dirBatch: [Float] = []
        let deltaValuesData = Data(bytes: &deltaValues, count: deltaValues.count*MemoryLayout<Float>.size)
        let mpsDevice = MPSGraphDevice(mtlDevice: device!)
        let deltaValuesTensorData = MPSGraphTensorData(device: mpsDevice, data: deltaValuesData, shape: [maxSteps as NSNumber], dataType: .float32)
        for i in 1...numEpochs {
            loss = 0.0
            var firstIndex = 0
            
            while (firstIndex < numLabels) {
                autoreleasepool {
                    inputBatch.removeAll()
                    labelBatch.removeAll()
                    dirBatch.removeAll()
                    let curBatchSize: Int = min(batchSize, numLabels-firstIndex)
                    for arr in inputDataShuffled[firstIndex..<firstIndex+curBatchSize] {
                        for elem in arr {
                            inputBatch.append(elem)
                        }
                    }
                    for arr in labelDataShuffled[firstIndex..<firstIndex+curBatchSize] {
                        for elem in arr {
                            labelBatch.append(elem)
                        }
                    }
                    for arr in dirDataShuffled[firstIndex..<firstIndex+curBatchSize] {
                        for elem in arr {
                            dirBatch.append(elem)
                        }
                    }
                    
                    doubleBufferingSemaphore.wait()
                    let inputTensorDataTemp = Data(bytes: &inputBatch, count: MemoryLayout<Float>.size*inputBatch.count)
                    let inputTensorDataCurrent = MPSGraphTensorData(device: mpsDevice, data: inputTensorDataTemp, shape: [curBatchSize as NSNumber, maxSteps as NSNumber, 3], dataType: inputDataType)
                    //print(inputTensorData[currentTensorDataIndex].shape)
                    let labelTensorDataTemp = Data(bytes: &labelBatch, count: MemoryLayout<Float>.size*labelBatch.count)
                    let labelTensorDataCurrent = MPSGraphTensorData(device: mpsDevice, data: labelTensorDataTemp, shape: [curBatchSize as NSNumber, labelData[0].count as NSNumber], dataType: .float32)
                    let dirTensorDataTemp = Data(bytes: &dirBatch, count: MemoryLayout<Float>.size*dirBatch.count)
                    let dirTensorDataCurrent = MPSGraphTensorData(device: mpsDevice, data: dirTensorDataTemp, shape: [curBatchSize as NSNumber, 3], dataType: .float32)
    //                var epoch = Float(i)
    //                let epochTensorDataTemp = Data(bytes: &epoch, count: MemoryLayout<Float>.size)
    //                epochTensorData[currentTensorDataIndex] = MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device!), data: epochTensorDataTemp, shape: [1], dataType: .float32)
                    //print(labelTensorData[currentTensorDataIndex].shape)
                    let execDescriptor = MPSGraphExecutionDescriptor()
                    execDescriptor.completionHandler = { (resultsDictionary, nil) in
                        self.doubleBufferingSemaphore.signal()
                        if (self.lossTensor != nil) {
                            let result = resultsDictionary[self.lossTensor!]
                            var loss: [Float] = Array(repeating: 0.0, count: 1)
                            result!.mpsndarray().readBytes(&loss, strideBytes: nil)
                            self.loss += loss[0]
                        }
                    }
                    graph.runAsync(with: commandQueue!, feeds: [inputsPlaceholderTensor: inputTensorDataCurrent, labelsPlaceholderTensor: labelTensorDataCurrent, cameraDirPlaceholderTensor!: dirTensorDataCurrent, deltaTPlaceholderTensor!: deltaValuesTensorData], targetTensors: targetTrainingTensors, targetOperations: targetTrainingOps, executionDescriptor: execDescriptor)
    //                graph.runAsync(with: commandQueue!, feeds: [inputsPlaceholderTensor: inputTensorData[currentTensorDataIndex], labelsPlaceholderTensor: labelTensorData[currentTensorDataIndex], epochPlaceholderTensor: epochTensorData[currentTensorDataIndex]], targetTensors: targetTrainingTensors, targetOperations: targetTrainingOps, executionDescriptor: execDescriptor)
                    
                    firstIndex += batchSize
                }
                
            }
            

            shuffledIndices.shuffle()
            inputDataShuffled = shuffledIndices.map{inputData[$0]}
            labelDataShuffled = shuffledIndices.map{labelData[$0]}
            print("Loss: \(loss)")
        }
            
    }
    func getInferenceDepthMapPiecemealDynamic(imageWidth: Int, imageHeight: Int, inputData: [[Float]], batchSize: Int, cameraPos: simd_float3, cameraAxis: simd_float3, cameraAngle: Float, focalLength: Float, maxT: Double, minWorldBound: simd_float3, maxWorldBound: simd_float3, getWeights: Bool, resultsOut: inout [MPSGraphTensor: MPSGraphTensorData]) -> ([Float], [Float]) {
        assert(inputData.count == imageWidth*imageHeight)
        assert(inputData[0].count == 2)
        let mpsDevice = MPSGraphDevice(mtlDevice: device!)
        var results: [Float] = []
        var depthResults: [Float] = []
        var deltaValues: [Float] = []
        var currentT: Float = 1.0
        for i in 0..<maxSteps {
            let b = exp( (log(maxT) - log(1.0)) / Double(maxSteps))
            let newT = Float(1.0*pow(b, Double(i+1)))
            let deltaT = newT-currentT
            deltaValues.append(deltaT)
            currentT = newT
            //print("t[\(i)]: \(currentT), deltaT[\(i)]: \(deltaT)")
        }
        let deltaValuesData = Data(bytes: &deltaValues, count: deltaValues.count*MemoryLayout<Float>.size)
        let deltaValuesTensorData = MPSGraphTensorData(device: mpsDevice, data: deltaValuesData, shape: [maxSteps as NSNumber], dataType: .float32)
        var firstIndex = 0
        while (firstIndex < inputData.count) {
            autoreleasepool {
                var inputBatch: [Float] = []
                var dirBatch: [Float] = []
                let curBatchSize: Int = min(batchSize, inputData.count-firstIndex)

                for arr in inputData[firstIndex..<firstIndex+curBatchSize] {
                    let rayDir = simd_float3(x: arr[0] - 0.5, y: arr[1] - 0.5, z: -focalLength)
                    let rayDirRot = rotateVector(dir: rayDir, axis: cameraAxis, angle: cameraAngle)
                    let raySteps = castRay(cameraDir: rayDirRot, cameraPos: cameraPos, numSteps: maxSteps, minWorldBound: minWorldBound, maxWorldBound: maxWorldBound, deltaValues: deltaValues)
                    inputBatch.append(contentsOf: raySteps)
                    let rayDirNormalized = simd_normalize(rayDir)
                    dirBatch.append(rayDirNormalized.x)
                    dirBatch.append(rayDirNormalized.y)
                    dirBatch.append(rayDirNormalized.z)
                }
                let inputTensorDataTemp = Data(bytes: &inputBatch, count: MemoryLayout<Float>.size*inputBatch.count)
                let inputTensorDataCurrent = MPSGraphTensorData(device: mpsDevice, data: inputTensorDataTemp, shape: [curBatchSize as NSNumber, maxSteps as NSNumber, 3], dataType: .float32)
                let dirTensorDataTemp = Data(bytes: &dirBatch, count: MemoryLayout<Float>.size*dirBatch.count)
                let dirTensorDataCurrent = MPSGraphTensorData(device: mpsDevice, data: dirTensorDataTemp, shape: [curBatchSize as NSNumber, 3], dataType: .float32)
                var targetTensors: [MPSGraphTensor] = [targetInferenceTensor!]
                if (getWeights) {
                    targetTensors.append(contentsOf: layerWeightTensors)
                    targetTensors.append(contentsOf: layerBiasTensors)
                    targetTensors.append(contentsOf: hashTableWeightTensors)
                }
                let resultDict = graph.run(with: commandQueue!, feeds: [inputsPlaceholderTensor: inputTensorDataCurrent, cameraDirPlaceholderTensor!: dirTensorDataCurrent, deltaTPlaceholderTensor!: deltaValuesTensorData], targetTensors: targetTensors, targetOperations: nil)
                let depthResultsDict = graph.run(with: commandQueue!, feeds: [inputsPlaceholderTensor: inputTensorDataCurrent, cameraDirPlaceholderTensor!: dirTensorDataCurrent, deltaTPlaceholderTensor!: deltaValuesTensorData], targetTensors: [depthMapTensor!], targetOperations: nil)
                resultsOut = resultDict
                var resultArr: [Float] = Array(repeating: 0.0, count: curBatchSize*self.labelSize)
                resultDict[targetInferenceTensor!]!.mpsndarray().readBytes(&resultArr, strideBytes: nil)
                var depthResultArr: [Float] = Array(repeating: 0.0, count: curBatchSize)
                depthResultsDict[depthMapTensor!]!.mpsndarray().readBytes(&depthResultArr, strideBytes: nil)
                results.append(contentsOf: resultArr)
                depthResults.append(contentsOf: depthResultArr)
            }
            firstIndex += batchSize
        }
        return (results, depthResults)
    }
    
    func trainDoubleBufferedDynamic(imageWidth: Int, imageHeight: Int, inputData: [[Float]], labelData: [[Float]], numLabels: Int, numEpochs: Int, batchSize: Int, cameraPos: simd_float3, cameraAxis: simd_float3, cameraAngle: Float, focalLength: Float, maxT: Double, minWorldBound: simd_float3, maxWorldBound: simd_float3) {
        assert(inputData.count == labelData.count)
        assert(inputData.count == numLabels)
        assert(inputData.count == imageWidth*imageHeight)
        assert(inputData[0].count == 2)
        assert(labelData[0].count == labelSize)
        var shuffledIndices = inputData.indices.shuffled()
        var inputDataShuffled = shuffledIndices.map{inputData[$0]}
        var labelDataShuffled = shuffledIndices.map{labelData[$0]}
        let mpsDevice = MPSGraphDevice(mtlDevice: device!)
        let execDescriptor = MPSGraphExecutionDescriptor()
        execDescriptor.completionHandler = { (resultsDictionary, nil) in
            self.doubleBufferingSemaphore.signal()
            if (self.lossTensor != nil) {
                let result = resultsDictionary[self.lossTensor!]
                var loss: [Float] = Array(repeating: 0.0, count: 1)
                result!.mpsndarray().readBytes(&loss, strideBytes: nil)
                self.loss += loss[0]
            }
        }
        var deltaValues: [Float] = []
        var currentT: Float = 1.0
        for i in 0..<maxSteps {
            let b = exp( (log(maxT) - log(1.0)) / Double(maxSteps))
            let newT = Float(1.0*pow(b, Double(i+1)))
            let deltaT = newT-currentT
            deltaValues.append(deltaT)
            currentT = newT
            print("t[\(i)]: \(currentT), deltaT[\(i)]: \(deltaT)")
        }
        let deltaValuesData = Data(bytes: &deltaValues, count: deltaValues.count*MemoryLayout<Float>.size)
        let deltaValuesTensorData = MPSGraphTensorData(device: mpsDevice, data: deltaValuesData, shape: [maxSteps as NSNumber], dataType: .float32)
        for _ in 1...numEpochs {
            loss = 0.0
            var firstIndex = 0
            while (firstIndex < numLabels) {
                autoreleasepool {
                    var inputBatch: [Float] = []
                    var labelBatch: [Float] = []
                    var dirBatch: [Float] = []
                    let curBatchSize: Int = min(batchSize, numLabels-firstIndex)

                    for arr in inputDataShuffled[firstIndex..<firstIndex+curBatchSize] {
                        let rayDir = simd_float3(x: arr[0] - 0.5, y: arr[1] - 0.5, z: -focalLength)
                        let rayDirRot = rotateVector(dir: rayDir, axis: cameraAxis, angle: cameraAngle)
                        let raySteps = castRay(cameraDir: rayDirRot, cameraPos: cameraPos, numSteps: maxSteps, minWorldBound: minWorldBound, maxWorldBound: maxWorldBound, deltaValues: deltaValues)
                        inputBatch.append(contentsOf: raySteps)
                        let rayDirNormalized = simd_normalize(rayDir)
                        dirBatch.append(rayDirNormalized.x)
                        dirBatch.append(rayDirNormalized.y)
                        dirBatch.append(rayDirNormalized.z)
                    }
                    
                    for arr in labelDataShuffled[firstIndex..<firstIndex+curBatchSize] {
                        for elem in arr {
                            labelBatch.append(elem)
                        }
                    }
                    
                    doubleBufferingSemaphore.wait()
                    let inputTensorDataTemp = Data(bytes: &inputBatch, count: MemoryLayout<Float>.size*inputBatch.count)
                    let inputTensorDataCurrent = MPSGraphTensorData(device: mpsDevice, data: inputTensorDataTemp, shape: [curBatchSize as NSNumber, maxSteps as NSNumber, 3], dataType: .float32)
                    //print(inputTensorData[currentTensorDataIndex].shape)
                    let labelTensorDataTemp = Data(bytes: &labelBatch, count: MemoryLayout<Float>.size*labelBatch.count)
                    let labelTensorDataCurrent = MPSGraphTensorData(device: mpsDevice, data: labelTensorDataTemp, shape: [curBatchSize as NSNumber, labelData[0].count as NSNumber], dataType: .float32)
                    let dirTensorDataTemp = Data(bytes: &dirBatch, count: MemoryLayout<Float>.size*dirBatch.count)
                    let dirTensorDataCurrent = MPSGraphTensorData(device: mpsDevice, data: dirTensorDataTemp, shape: [curBatchSize as NSNumber, 3], dataType: .float32)
                    graph.runAsync(with: commandQueue!, feeds: [inputsPlaceholderTensor: inputTensorDataCurrent, labelsPlaceholderTensor: labelTensorDataCurrent, cameraDirPlaceholderTensor!: dirTensorDataCurrent, deltaTPlaceholderTensor!: deltaValuesTensorData], targetTensors: targetTrainingTensors, targetOperations: targetTrainingOps, executionDescriptor: execDescriptor)
                }
                firstIndex += batchSize
            }
            shuffledIndices.shuffle()
            inputDataShuffled = shuffledIndices.map{inputData[$0]}
            labelDataShuffled = shuffledIndices.map{labelData[$0]}
            print("Loss: \(loss)")
        }
        
    }
    
    func generateDepthMap(inputData: [[Float]], cameraPos: simd_float3, cameraAxis: simd_float3, cameraAngle: Float, focalLength: Float, maxT: Double) -> [Float] {
        let mps_device = MPSGraphDevice(mtlDevice: device!)
        var depthPosInputs: [Float] = []
        for arr in inputData {
            let rayDir = simd_float3(x: arr[0] - 0.5, y: arr[1] - 0.5, z: -focalLength)
            let rayDirRot = rotateVector(dir: rayDir, axis: cameraAxis, angle: cameraAngle)
            depthPosInputs.append(rayDirRot.x)
            depthPosInputs.append(rayDirRot.y)
            depthPosInputs.append(rayDirRot.z)
        }
        let depthPosInputsData = Data(bytes: &depthPosInputs, count: MemoryLayout<Float>.size*depthPosInputs.count)
        //let depthPosInputsTensorData = MPSGraphTensorData(device: mps_device, data: depthPosInputsData, shape: [inputData.count as NSNumber, 2], dataType: .float32)
        let depthPosTensor = graph.constant(depthPosInputsData, shape: [inputData.count as NSNumber, 3], dataType: .float32)
        var cameraPosArr: [Float] = [cameraPos.x, cameraPos.y, cameraPos.z]
        let cameraPosData = Data(bytes: &cameraPosArr, count: MemoryLayout<Float>.size*3)
        let cameraPosTensor = graph.constant(cameraPosData, shape: [3], dataType: .float32)
        //let cameraPosTensorData = MPSGraphTensorData(device: mps_device, data: cameraPosData, shape: [3], dataType: .float32)
        var depthMapTensor = getDepthMapTensor(deltaStepScale: 0.3, rayThreshold: -log(0.01), maxT: maxT, depthInputTensor: depthPosTensor, cameraPosTensor: cameraPosTensor )
        let resultsDict = graph.run(feeds: [:], targetTensors: [depthMapTensor], targetOperations: nil)
        let resultsData = resultsDict[depthMapTensor]
        var results: [Float] = []
        if (resultsData != nil) {
            results = Array(repeating: 0.0, count: inputData.count)
            resultsData!.mpsndarray().readBytes(&results, strideBytes: nil)
        }
        return results
    }
    
    func generateDepthMap2(inputData: [[Float]], dirData: [[Float]], deltaValues: inout [Float]) -> [Float] {
        autoreleasepool {
            let mpsDevice = MPSGraphDevice(mtlDevice: device!)
            var inputDataFlattened: [Float] = Array(inputData.joined())
            var dirDataFlattened: [Float] = Array(dirData.joined())
            let inputTensorData = Data(bytes: &inputDataFlattened, count: MemoryLayout<Float>.size*inputDataFlattened.count)
            let inputGraphTensorData = MPSGraphTensorData(device: mpsDevice, data: inputTensorData, shape: [inputData.count as NSNumber, maxSteps as NSNumber, 3], dataType: .float32)
            let dirTensorData = Data(bytes: &dirDataFlattened, count: MemoryLayout<Float>.size*dirDataFlattened.count)
            let dirGraphTensorData = MPSGraphTensorData(device: mpsDevice, data: dirTensorData, shape: [dirData.count as NSNumber, dirData[0].count as NSNumber], dataType: .float32)
            let deltaTensorData = Data(bytes: &deltaValues, count: MemoryLayout<Float>.size*deltaValues.count)
            let deltaGraphTensorData = MPSGraphTensorData(device: mpsDevice, data: deltaTensorData, shape: [deltaValues.count as NSNumber], dataType: .float32)
            let resultsDict = graph.run(feeds: [inputsPlaceholderTensor : inputGraphTensorData, cameraDirPlaceholderTensor!: dirGraphTensorData, deltaTPlaceholderTensor!: deltaGraphTensorData], targetTensors: [depthMapTensor!], targetOperations: nil)
            var results: [Float] = []
            let resultsData = resultsDict[depthMapTensor!]
            if (resultsData != nil) {
                results = Array(repeating: 0, count: inputData.count)
                resultsData!.mpsndarray().readBytes(&results, strideBytes: nil)
            }
            return results
        }

    }
    
    func getInference(inputData: [[Float]], dirData: [[Float]], deltaValues: inout [Float], getWeights: Bool, resultsOut: inout [MPSGraphTensor: MPSGraphTensorData]) -> [Float] {
        var results: [Float] = []
        if (targetInferenceTensor != nil) {
            let mpsDevice = MPSGraphDevice(mtlDevice: device!)
            var inputDataFlattened: [Float] = Array(inputData.joined())
            var dirDataFlattened: [Float] = Array(dirData.joined())
            let inputTensorData = Data(bytes: &inputDataFlattened, count: MemoryLayout<Float>.size*inputDataFlattened.count)
            let inputGraphTensorData = MPSGraphTensorData(device: mpsDevice, data: inputTensorData, shape: [inputData.count as NSNumber, maxSteps as NSNumber, 3], dataType: .float32)
            let dirTensorData = Data(bytes: &dirDataFlattened, count: MemoryLayout<Float>.size*dirDataFlattened.count)
            let dirGraphTensorData = MPSGraphTensorData(device: mpsDevice, data: dirTensorData, shape: [dirData.count as NSNumber, dirData[0].count as NSNumber], dataType: .float32)
            let deltaTensorData = Data(bytes: &deltaValues, count: MemoryLayout<Float>.size*deltaValues.count)
            let deltaGraphTensorData = MPSGraphTensorData(device: mpsDevice, data: deltaTensorData, shape: [deltaValues.count as NSNumber], dataType: .float32)
            var targetTensors: [MPSGraphTensor] = [targetInferenceTensor!]
            if (getWeights) {
                targetTensors.append(contentsOf: layerWeightTensors)
                targetTensors.append(contentsOf: layerBiasTensors)
                targetTensors.append(contentsOf: hashTableWeightTensors)
            }
            let resultsDict = graph.run(feeds: [inputsPlaceholderTensor : inputGraphTensorData, cameraDirPlaceholderTensor!: dirGraphTensorData, deltaTPlaceholderTensor!: deltaGraphTensorData], targetTensors: targetTensors, targetOperations: nil)
            resultsOut = resultsDict
            let resultsData = resultsDict[targetInferenceTensor!]
            if (resultsData != nil) {
                results = Array(repeating: 0.0, count: inputData.count*labelSize)
                resultsData!.mpsndarray().readBytes(&results, strideBytes: nil)
            }
            
        }
        return results
    }
}
