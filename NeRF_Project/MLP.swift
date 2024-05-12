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
    var layers: [Layer] = []
    var layerActivationFunctions: [ActivationFunction] = []
    var layerWeightTensors: [MPSGraphTensor] = []
    var layerBiasTensors: [MPSGraphTensor] = []
    var variableTensors: [MPSGraphTensor] = []
    var device: MTLDevice?
    var commandQueue: MTLCommandQueue?
    var doubleBufferingSemaphore: DispatchSemaphore = DispatchSemaphore(value: 2)
    var inputShape: [NSNumber]
    var labelSize: Int
    var graph: MPSGraph
    var lossTensor: MPSGraphTensor?
    var inputsPlaceholderTensor: MPSGraphTensor
    var labelsPlaceholderTensor: MPSGraphTensor
    var targetTrainingTensors: [MPSGraphTensor] = []
    var targetInferenceTensor: MPSGraphTensor?
    var targetTrainingOps: [MPSGraphOperation] = []
    var loss: Float = 0.0
    //inputShape: The dimensions of the tensor used as an input
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
        var biasTensor: MPSGraphTensor
        if (initialBiasArray == nil) {
            biasTensor = graph.variable(with: layer.biasData, shape: [outputSize as NSNumber], dataType: .float32, name: nil)
        } else {
            var initial = initialBiasArray!
            let data = Data(bytes: &initial, count: MemoryLayout<Float>.size*initial.count)
            biasTensor = graph.variable(with: data, shape: [outputSize as NSNumber], dataType: .float32, name: nil)
        }
        layerWeightTensors.append(weightTensor)
        variableTensors.append(weightTensor)
        layerBiasTensors.append(biasTensor)
        variableTensors.append(biasTensor)
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
        for variableTensor in variableTensors {
            let updateTensor = graph.stochasticGradientDescent(learningRate: graph.constant(0.001, dataType: .float32), values: variableTensor, gradient: gradientTensors[variableTensor]!, name: nil)
            let assignOp = graph.assign(variableTensor, tensor: updateTensor, name: nil)
            targetTrainingOps.append(assignOp)
        }
    }
    func trainDoubleBuffered(inputData: [[Float]], labelData: [[Float]], numLabels: Int, numEpochs: Int, batchSize: Int, inputDataType: MPSDataType) {
        assert(inputData.count == labelData.count)
        var shuffledIndices = inputData.indices.shuffled()
        var inputDataShuffled = shuffledIndices.map{inputData[$0]}
        var labelDataShuffled = shuffledIndices.map{labelData[$0]}
        
        var inputBatch: [Float] = []
        var labelBatch: [Float] = []
        for _ in 1...numEpochs {
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
        let weights = CodableWeights()
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
    private func hashTensor(indexTensor: MPSGraphTensor, primeTensor: MPSGraphTensor, hashTableSizeTensor: MPSGraphTensor, corner: Corner2D, dimension: Int) -> MPSGraphTensor {
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
        let yScaledTensor = graph.multiplication(yIndexTensor, primeTensor, name: nil)
        let xorTensor = graph.bitwiseXOR(xIndexTensor, yScaledTensor, name: nil)
        let moduloTensor = graph.modulo(xorTensor, hashTableSizeTensor, name: nil)
        return moduloTensor
    }
    func invokeHashTable(inputTensor: MPSGraphTensor?, dimension: Int) -> MPSGraphTensor {
        assert(hashTableInitialized)
        var inTensor: MPSGraphTensor
        if (inputTensor == nil) {
            inTensor = inputsPlaceholderTensor
        } else {
            inTensor = inputTensor!
        }
        let prime1Data = Data(bytes: &prime1, count: MemoryLayout<Int>.size)
        let prime1Tensor = graph.constant(prime1Data, shape: [1], dataType: .int32)
        var interpolatedWeightTensors: [MPSGraphTensor] = []
        for i in 0..<numLevels {
            let resolutionFloatTensor = graph.cast(resolutionTensors[i], to: .float32, name: nil)
            let scaledInputTensor = graph.multiplication(resolutionFloatTensor, inTensor, name: nil)
            let lowerLeftTensor = graph.floor(with: scaledInputTensor, name: nil)
            let lowerLeftIndexTensor = graph.cast(lowerLeftTensor, to: .int32, name: nil)
            let lowerLeftHashTensor = hashTensor(indexTensor: lowerLeftIndexTensor, primeTensor: prime1Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .ll, dimension: dimension)
            var lowerLeftWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: lowerLeftHashTensor, batchDimensions: 0, name: nil)
            let lowerRightHashTensor = hashTensor(indexTensor: lowerLeftIndexTensor, primeTensor: prime1Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .lr, dimension: dimension)
            let upperLeftHashTensor = hashTensor(indexTensor: lowerLeftIndexTensor, primeTensor: prime1Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .ul, dimension: dimension)
            let upperRightHashTensor = hashTensor(indexTensor: lowerLeftIndexTensor, primeTensor: prime1Tensor, hashTableSizeTensor: hashTableSizeTensors[i], corner: .ur, dimension: dimension)
            var lowerRightWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: lowerRightHashTensor, batchDimensions: 0, name: nil)
            var upperLeftWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: upperLeftHashTensor, batchDimensions: 0, name: nil)
            var upperRightWeightTensor = graph.gatherND(withUpdatesTensor: hashTableWeightTensors[i], indicesTensor: upperRightHashTensor, batchDimensions: 0, name: nil)
            let onesTensor = graph.constant(1.0, dataType: .float32)
            let lowerLeftDiffTensor = graph.subtraction(scaledInputTensor, lowerLeftTensor, name: nil)
            let oneMinusLowerLeftDiffTensor = graph.subtraction(onesTensor, lowerLeftDiffTensor, name: nil)
            let lowerLeftDiffXTensor = graph.sliceTensor(lowerLeftDiffTensor, dimension: dimension, start: 0, length: 1, name: nil)
            let lowerLeftDiffYTensor = graph.sliceTensor(lowerLeftDiffTensor, dimension: dimension, start: 1, length: 1, name: nil)
            let oneMinusLowerLeftDiffXTensor = graph.sliceTensor(oneMinusLowerLeftDiffTensor, dimension: dimension, start: 0, length: 1, name: nil)
            let oneMinusLowerLeftDiffYTensor = graph.sliceTensor(oneMinusLowerLeftDiffTensor, dimension: dimension, start: 1, length: 1, name: nil)
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
            //print("Level \(i): \(hashTableSize) elements")
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
            hashTableWeightTensors.append(weightTensor)
            variableTensors.append(weightTensor)
        }
        hashTableInitialized = true
    }
    override func getVariableData(results: [MPSGraphTensor: MPSGraphTensorData]) -> Data {
        let weights = CodableWeights()
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
            hashTableWeightTensors.append(weightTensor)
            variableTensors.append(weightTensor)
        }
        hashTableInitialized = true
    }
    
    override func invokeHashTable(inputTensor: MPSGraphTensor?, dimension: Int) -> MPSGraphTensor {
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
        return activationTensor
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
        }
        activationTensor = invokeMLP(inputTensor: activationTensor)
        let densityTensor = graph.sliceTensor(activationTensor, dimension: 2, start: 0, length: 1, name: nil)
        let colorTensor = graph.sliceTensor(activationTensor, dimension: 2, start: 1, length: 3, name: nil)
        let deltaReshapeTensor = graph.reshape(deltaTPlaceholderTensor!, shape: [1,maxSteps as NSNumber, 1], name: nil)
        let scaledDensityTensor = graph.multiplication(densityTensor, deltaReshapeTensor, name: nil)
        var transmittanceTensor = graph.cumulativeSum(scaledDensityTensor, axis: 1, exclusive: true, reverse: false, name: nil)
        transmittanceTensor = graph.negative(with: scaledDensityTensor, name: nil)
        transmittanceTensor = graph.exponent(with: transmittanceTensor, name: nil)
        var colorScaleTensor = graph.negative(with: scaledDensityTensor, name: nil)
        colorScaleTensor = graph.exponent(with: colorScaleTensor, name: nil)
        colorScaleTensor = graph.subtraction(graph.constant(1.0, dataType: .float32), colorScaleTensor, name: nil)
        colorScaleTensor = graph.multiplication(transmittanceTensor, colorScaleTensor, name: nil)
        let unsummedColorsTensor = graph.multiplication(colorScaleTensor, colorTensor, name: nil)
        var approxColorTensor = graph.reductionSum(with: unsummedColorsTensor, axis: 1, name: nil)
        approxColorTensor = graph.reshape(approxColorTensor, shape: [-1, 3], name: nil)
        targetInferenceTensor = approxColorTensor
        depthMapTensor = getDepthMapTensor()
        lossTensor = graph.subtraction(approxColorTensor, labelsPlaceholderTensor, name: nil)
        lossTensor = graph.multiplication(lossTensor!, lossTensor!, name: nil)
        lossTensor = graph.reductionSum(with: lossTensor!, axis: 0, name: nil)
        lossTensor = graph.reductionSum(with: lossTensor!, axis: 1, name: nil)
        targetTrainingTensors = [lossTensor!]
        let gradientTensors = graph.gradients(of: lossTensor!, with: variableTensors, name: nil)
        for variableTensor in variableTensors {
            let updateTensor = graph.stochasticGradientDescent(learningRate: graph.constant(0.001, dataType: .float32), values: variableTensor, gradient: gradientTensors[variableTensor]!, name: nil)
            let assignOp = graph.assign(variableTensor, tensor: updateTensor, name: nil)
            targetTrainingOps.append(assignOp)
        }
    }
    func getDepthMapTensor() -> MPSGraphTensor {
        var activationTensor: MPSGraphTensor = inputsPlaceholderTensor
        if (hashTableInitialized) {
            activationTensor = invokeHashTable(inputTensor: nil, dimension: 2)
        }
        activationTensor = invokeMLP(inputTensor: activationTensor)
        let densityTensor = graph.sliceTensor(activationTensor, dimension: 2, start: 0, length: 1, name: nil)
        let deltaReshapeTensor = graph.reshape(deltaTPlaceholderTensor!, shape: [1,maxSteps as NSNumber, 1], name: nil)
        let scaledDensityTensor = graph.multiplication(densityTensor, deltaReshapeTensor, name: nil)
        var sumTensor = graph.reductionSum(with: scaledDensityTensor, axis: 1, name: nil)
        sumTensor = graph.negative(with: sumTensor, name: nil)
        sumTensor = graph.exponent(with: sumTensor, name: nil)
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
        let dirDataShuffled = shuffledIndices.map{dirData[$0]}
        var inputBatch: [Float] = []
        var labelBatch: [Float] = []
        var dirBatch: [Float] = []
        let deltaValuesData = Data(bytes: &deltaValues, count: deltaValues.count*MemoryLayout<Float>.size)
        let mpsDevice = MPSGraphDevice(mtlDevice: device!)
        let deltaValuesTensorData = MPSGraphTensorData(device: mpsDevice, data: deltaValuesData, shape: [maxSteps as NSNumber], dataType: .float32)
        for _ in 1...numEpochs {
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
                    
                    firstIndex += batchSize
                }
                
            }
            

            shuffledIndices.shuffle()
            inputDataShuffled = shuffledIndices.map{inputData[$0]}
            labelDataShuffled = shuffledIndices.map{labelData[$0]}
            print("Loss: \(loss)")
        }
            
    }
    func getInferenceDepthMap(imageWidth: Int, imageHeight: Int, inputData: [[Float]], batchSize: Int, cameraPos: simd_float3, cameraAxis: simd_float3, cameraAngle: Float, focalLength: Float, maxT: Double, minWorldBound: simd_float3, maxWorldBound: simd_float3, getWeights: Bool, resultsOut: inout [MPSGraphTensor: MPSGraphTensorData]) -> ([Float], [Float]) {
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
    
}
