//
//  main.swift
//  mpsgraph_test
//
//  Created by Josh Lakin on 3/7/24.
//

import Foundation
import UniformTypeIdentifiers
import CoreGraphics
import ImageIO
import Metal
import MetalPerformanceShadersGraph

let maxT: Double = 8
let imageIndex = 0
//let (cameraAngle, cameraPos) = generateCameraPosAngleCircleImage(imageIndex: imageIndex, numImages: 8, radius: 8.0, center: simd_float3(0.0, 0.0, 0.0))
//let (cameraAngle, cameraPos) = generateCameraPosAngleCircle(scale: 0.0, radius: 8.0, center: simd_float3(0.0, 0.0, 0.0))
//print("Camera angle: \(cameraAngle), Camera Pos: [\(cameraPos.x),\(cameraPos.y),\(cameraPos.z)]")
let cameraPos: [simd_float3] = [simd_float3(x: -0.8, y: 0, z: -7.916),
                                simd_float3(x: 0, y: 0, z: -8.0),
                                simd_float3(x: 0.8, y: 0, z: -7.916)]
let cameraAngle: [Float] = [-0.100719, 0, 0.100719]
let maxSteps = 16
let outFilePath = URL(fileURLWithPath: "weights.json")
let filePath = URL(fileURLWithPath: "testimage\(imageIndex+1).png")
let cgiSource = CGImageSourceCreateWithURL(filePath as CFURL, [:] as CFDictionary)
var cgi: CGImage?
var imageData: [Float] = []
var imageData2d: [[Float]] = []
var inputs: [[Float]] = []
if (cgiSource != nil) {
    print("Got image")
    cgi = CGImageSourceCreateImageAtIndex(cgiSource!, 0, nil)
}
var cameraDirs: [[Float]] = []
var deltaValues: [Float] = []
if (cgi != nil) {
    //var stride = 3
    //print(cgi!.bitsPerPixel)
    imageData = convertByteImageToFloat(image: cgi!)
    imageData2d = convertArray1DTo2D(arr: imageData, stride: 3, count: cgi!.width*cgi!.height)
    //checkArrayEqual(arr: imageData, stride: 3, count: cgi!.width*cgi!.height, arr2d: imageData2d)
//    (inputBatches, dirBatches, deltaT, labelBatches) = createBatchesNeRFBad(imageWidth: cgi!.width, imageHeight: cgi!.height, batchSize: 64, floatImage: imageData2d, numSteps: 1, stepDelta: 0.8, cameraPos: simd_float3(x: 0.0, y: 0.0, z: 0.0), cameraAxis: simd_float3(x: 0.0, y: 0.0, z: 1.0), cameraAngle: 0.0, minWorldBound: simd_float3(x: -10, y: -10, z: -10), maxWorldBound: simd_float3(x: 10, y: 10, z: 10))
    //(inputs, cameraDirs, deltaValues) = createNeRFInputData(originalImage: cgi!, rowStart: 0, colStart: 0, blockWidth: cgi!.width, blockHeight: cgi!.height, numSteps: 25, maxT: 8.0, cameraFocalLength: 0.5, cameraPos: simd_float3(0,0,0), cameraAxis: simd_float3(0,1,0), cameraAngle: 0, minWorldBound: simd_float3(-10,-10,-10), maxWorldBound: simd_float3(10,10,10))
    //inputs = createInputLabels3D(originalImage: cgi!)
    inputs = createInputLabels(imageWidth: cgi!.width, imageHeight: cgi!.height)
    print("Copied data")
    //print(cgi!.bitsPerComponent)
    //inputs = createInputLabels(originalImage: cgi!)
    
}

var retrievedWeights: Bool = false
if (!imageData.isEmpty) {
    var weights: CodableWeights? = nil
    do {
        let data = try Data(contentsOf: outFilePath, options: .mappedIfSafe)
        let jsonDecoder = JSONDecoder()
        weights = try jsonDecoder.decode(CodableWeights.self, from: data)
        retrievedWeights = true
    } catch {
        print("weights.json not found")
    }
   
    let nerf: NeRF = NeRF(labelSize: 3, numTrainingImages: 1, maxSteps: maxSteps, maxT: maxT)
    if (retrievedWeights) {
        nerf.createHashTable(minResolution: 16, maxResolution: 1024, maxEntries: 65536, initialHashTableWeights: weights!.hashTableWeights)
        nerf.addDenseLayer(inputSize: 32, outputSize: 64, activationFunction: .ReLU, initialWeightArray: weights!.weightMatrices[0], initialBiasArray: weights!.biasVectors[0])
        nerf.addDenseLayer(inputSize: 64, outputSize: 64, activationFunction: .ReLU, initialWeightArray: weights!.weightMatrices[1], initialBiasArray: weights!.biasVectors[1])
        nerf.addDenseLayer(inputSize: 64, outputSize: 4, activationFunction: .Sigmoid, initialWeightArray: weights!.weightMatrices[2], initialBiasArray: weights!.biasVectors[2])
    } else {
        nerf.createHashTable(minResolution: 16, maxResolution: 1024, maxEntries: 65536, initialHashTableWeights: nil)
        nerf.addDenseLayer(inputSize: 32, outputSize: 64, activationFunction: .ReLU, initialWeightArray: nil, initialBiasArray: nil)
        nerf.addDenseLayer(inputSize: 64, outputSize: 64, activationFunction: .ReLU, initialWeightArray: nil, initialBiasArray: nil)
        nerf.addDenseLayer(inputSize: 64, outputSize: 4, activationFunction: .Sigmoid, initialWeightArray: nil, initialBiasArray: nil)
    }

    nerf.finalize(inputTensor: nil)
    //nerf.trainDoubleBuffered(inputData: inputs, labelData: imageData2d, dirData: cameraDirs, deltaValues: &deltaValues, numLabels: cgi!.width*cgi!.height, numEpochs: 3, batchSize: min(256, cgi!.width*cgi!.height), inputDataType: .float32)
    nerf.trainDoubleBufferedDynamic(imageWidth: cgi!.width, imageHeight: cgi!.height, inputData: inputs, labelData: imageData2d, numLabels: cgi!.width*cgi!.height, numEpochs: 3, batchSize: min(256, cgi!.width*cgi!.height), cameraPos: cameraPos[imageIndex], cameraAxis: simd_float3(0,1,0), cameraAngle: cameraAngle[imageIndex], focalLength: 0.5, maxT: maxT, minWorldBound: simd_float3(-10,-10,-10), maxWorldBound: simd_float3(10,10,10))
    var resultsDict: [MPSGraphTensor : MPSGraphTensorData] = [:]
    let (results, depthMap) = nerf.getInferenceDepthMapPiecemealDynamic(imageWidth: cgi!.width, imageHeight: cgi!.height, inputData: inputs, batchSize: cgi!.width*cgi!.height / 32, cameraPos: cameraPos[imageIndex], cameraAxis: simd_float3(0,1,0), cameraAngle: cameraAngle[imageIndex], focalLength: 0.5, maxT: maxT, minWorldBound: simd_float3(-10,-10,-10), maxWorldBound: simd_float3(10,10,10), getWeights: true, resultsOut: &resultsDict)
    //let depthMapRGB = convertDepthImageToRGB(depthImage: depthMap, numSteps: maxSteps)
    let depthMapRGB = convertDepthImageToRGB2(depthImage: depthMap)
    let jsonData = nerf.getVariableData(results: resultsDict)
    //mlp.createHashTable(minResolution: 16, maxResolution: 512, maxEntries: 65536)
//    if (retrievedWeights) {
//        mlp.createHashTable(minResolution: 16, maxResolution: 512, maxEntries: 65536, initialHashTableWeights: weights!.hashTableWeights)
//        mlp.addDenseLayer(inputSize: 32, outputSize: 64, activationFunction: .ReLU, initialWeightArray: weights!.weightMatrices[0], initialBiasArray: weights!.biasVectors[0])
//        mlp.addDenseLayer(inputSize: 64, outputSize: 64, activationFunction: .ReLU, initialWeightArray: weights!.weightMatrices[1], initialBiasArray: weights!.biasVectors[1])
//        mlp.addDenseLayer(inputSize: 64, outputSize: 3, activationFunction: .Sigmoid, initialWeightArray: weights!.weightMatrices[2], initialBiasArray: weights!.biasVectors[2])
//    } else {
//        mlp.createHashTable(minResolution: 16, maxResolution: 512, maxEntries: 65536, initialHashTableWeights: nil)
//        mlp.addDenseLayer(inputSize: 32, outputSize: 64, activationFunction: .ReLU, initialWeightArray: nil, initialBiasArray: nil)
//        mlp.addDenseLayer(inputSize: 64, outputSize: 64, activationFunction: .ReLU, initialWeightArray: nil, initialBiasArray: nil)
//        mlp.addDenseLayer(inputSize: 64, outputSize: 3, activationFunction: .Sigmoid, initialWeightArray: nil, initialBiasArray: nil)
//    }
//
//    let inputTensor = mlp.invokeHashTable()
//    mlp.finalize(inputTensor: inputTensor)
    //mlp.trainDoubleBuffered(inputData: inputs, labelData: imageData2d, numLabels: cgi!.width*cgi!.height, numEpochs: 5, batchSize: min(256, cgi!.width*cgi!.height), inputDataType: .float32)
//    let newInputs = createInputLabels(imageWidth: cgi!.width, imageHeight: cgi!.height)
//    let results = mlp.getInference(inputData: inputs, getWeights: true, resultsOut: &resultsDict)
//    let jsonData = mlp.getVariableData(results: resultsDict)
    try! jsonData.write(to: outFilePath)
//    let nerf: NeRF_Bad = NeRF_Bad(maxRaySteps: 1, minResolution: 16, maxResolution: 1024, maxEntries: 65536, numTrainingImages: 1, imageWidth: cgi!.width, imageHeight: cgi!.height, sceneBoundsLength: 20)
//    nerf.trainOnImage(imageIndex: 0, inputBatches: inputBatches, labelBatches: labelBatches, dirBatches: dirBatches, deltaValues: &deltaT, numEpochs: 2)
//    var raySteps: [Float] = inputBatches.flatMap { $0 }
//    var rayDirs: [Float] = dirBatches.flatMap { $0 }
//    let results = nerf.generateTestImage(imageIndex: 0, imageWidth: cgi!.width, imageHeight: cgi!.height, raySteps: &raySteps, deltaValues: &deltaT, rayDirs: &rayDirs)
    

    assert(results.count == 3*cgi!.width*cgi!.height)
    let outputPath = URL(fileURLWithPath: "output.png")
    //let depthOutputPath = URL(fileURLWithPath: "output_depth.png")
    let outImage = convertFloatImageToByte(floatImage: results, originalImage: cgi!, imageWidth: cgi!.width, imageHeight: cgi!.height)
//    //let depthOutImage = convertFloatImageToByte(floatImage: depthMapRGB, originalImage: cgi! )
    let imageDestination = CGImageDestinationCreateWithURL(outputPath as CFURL, UTType.png.identifier as CFString, 1, nil)
//    //let imageDestinationDepth = CGImageDestinationCreateWithURL(depthOutputPath as CFURL, UTType.png.identifier as CFString, 1, nil)
    if (imageDestination != nil && outImage != nil) {
        CGImageDestinationAddImage(imageDestination!, outImage!, nil)
        CGImageDestinationFinalize(imageDestination!)
        print("Wrote file to output.png")
    } else {
        print("Error writing file")
    }
//    if (imageDestinationDepth != nil && depthOutImage != nil) {
//        CGImageDestinationAddImage(imageDestinationDepth!, depthOutImage!, nil)
//        CGImageDestinationFinalize(imageDestinationDepth!)
//        print("Wrote file to output_depth.png")
//    } else {
//        print("Error writing file")
//    }
    
    
}
