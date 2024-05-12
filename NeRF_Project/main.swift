//
//  main.swift
//  NeRF_Project
//
//  Created by Josh Lakin on 5/10/24.
//

import Foundation

import Foundation
import UniformTypeIdentifiers
import CoreGraphics
import ImageIO
import Metal
import MetalPerformanceShadersGraph

let outFilePath = URL(fileURLWithPath: "weights.json")

let maxT: Double = 8
let maxSteps = 64
let numImages = 3
let epochs = 1

let cameraPos: [simd_float3] = [simd_float3(x: -0.8, y: 0, z: -7.916),
                                simd_float3(x: 0, y: 0, z: -8.0),
                                simd_float3(x: 0.8, y: 0, z: -7.916)]
let cameraAngle: [Float] = [-0.100719, 0, 0.100719]

let newCameraPos = simd_float3(x: -0.4, y: 0, z: -7.999)
let newCameraAngle: Float = -0.0499646309

var labels: [ [[Float]] ] = []
var lastCGI: CGImage? = nil
var inputs: [[Float]] = []
for imageIndex in 0..<numImages {
    let filePath = URL(fileURLWithPath: "testimage1\(imageIndex+1).png")

    let cgiSource = CGImageSourceCreateWithURL(filePath as CFURL, [:] as CFDictionary)
    var cgi: CGImage?
    var imageData: [Float] = []
    var imageData2d: [[Float]] = []
    if (cgiSource != nil) {
        print("Got image")
        cgi = CGImageSourceCreateImageAtIndex(cgiSource!, 0, nil)
    }
    if (cgi != nil) {
        imageData = convertByteImageToFloat(image: cgi!)
        imageData2d = convertArray1DTo2D(arr: imageData, stride: 3, count: cgi!.width*cgi!.height)
        if (imageIndex == 0) {
            inputs = createInputLabels(imageWidth: cgi!.width, imageHeight: cgi!.height)
        } else {
            assert(cgi!.width == lastCGI!.width && cgi!.height == lastCGI!.height)
        }
        labels.append(imageData2d)
        lastCGI = cgi!
        print("Copied data")
    }
}

var retrievedWeights: Bool = false
if (!labels.isEmpty) {
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
    for _ in 0..<epochs {
        for imageIndex in 0..<numImages {
            nerf.trainDoubleBufferedDynamic(imageWidth: lastCGI!.width, imageHeight: lastCGI!.height, inputData: inputs, labelData: labels[imageIndex], numLabels: lastCGI!.width*lastCGI!.height, numEpochs: 1, batchSize: min(256, lastCGI!.width*lastCGI!.height), cameraPos: cameraPos[imageIndex], cameraAxis: simd_float3(0,1,0), cameraAngle: cameraAngle[imageIndex], focalLength: 0.5, maxT: maxT, minWorldBound: simd_float3(-10,-10,-10), maxWorldBound: simd_float3(10,10,10))
        }
    }
    var resultsDict: [MPSGraphTensor: MPSGraphTensorData] = [:]
    let (results, depthMap) = nerf.getInferenceDepthMap(imageWidth: lastCGI!.width, imageHeight: lastCGI!.height, inputData: inputs, batchSize: lastCGI!.width*lastCGI!.height / 32, cameraPos: cameraPos[1], cameraAxis: simd_float3(0,1,0), cameraAngle: cameraAngle[1], focalLength: 0.5, maxT: maxT, minWorldBound: simd_float3(-10,-10,-10), maxWorldBound: simd_float3(10,10,10), getWeights: true, resultsOut: &resultsDict)
    let depthMapRGB = convertDepthImageToRGB2(depthImage: depthMap)
    //let depthMapRGB = convertDepthImageToRGB2(depthImage: depthMap)
    let jsonData = nerf.getVariableData(results: resultsDict)
    try! jsonData.write(to: outFilePath)

    assert(results.count == 3*lastCGI!.width*lastCGI!.height)
    let outputPath = URL(fileURLWithPath: "output.png")
    let depthOutputPath = URL(fileURLWithPath: "output_depth.png")
    let outImage = convertFloatImageToByte(floatImage: results, originalImage: lastCGI!, imageWidth: lastCGI!.width, imageHeight: lastCGI!.height)
    let depthOutImage = convertFloatImageToByte(floatImage: depthMapRGB, originalImage: lastCGI!, imageWidth: lastCGI!.width, imageHeight: lastCGI!.height )
    let imageDestination = CGImageDestinationCreateWithURL(outputPath as CFURL, UTType.png.identifier as CFString, 1, nil)
    let imageDestinationDepth = CGImageDestinationCreateWithURL(depthOutputPath as CFURL, UTType.png.identifier as CFString, 1, nil)
    if (imageDestination != nil && outImage != nil) {
        CGImageDestinationAddImage(imageDestination!, outImage!, nil)
        CGImageDestinationFinalize(imageDestination!)
        print("Wrote file to output.png")
    } else {
        print("Error writing file")
    }
    if (imageDestinationDepth != nil && depthOutImage != nil) {
        CGImageDestinationAddImage(imageDestinationDepth!, depthOutImage!, nil)
        CGImageDestinationFinalize(imageDestinationDepth!)
        print("Wrote file to output_depth.png")
    } else {
        print("Error writing file")
    }
    
    
}

