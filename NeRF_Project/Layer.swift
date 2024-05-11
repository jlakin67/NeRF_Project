//
//  Layer.swift
//  mpsgraph_test
//
//  Created by Josh Lakin on 3/14/24.
//

import Foundation
import Metal
import MetalPerformanceShadersGraph
import GameplayKit

func gaussianDist(mean: Float, std: Float) -> Float {
    let random = GKMersenneTwisterRandomSource()
    let a = Float(random.nextUniform())
    let b = Float(random.nextUniform())
    let z = sqrtf(-2 * logf(a)) * cosf(2 * .pi * b)

    return z * std + mean
}

func glorotInit(numInputs: Int, numOutputs: Int, isBias: Bool) -> [Float] {
    var result: [Float] = []
    let variance: Float = 2.0 / (Float(numInputs) + Float(numOutputs))
    let std = sqrtf(variance)
    if (isBias) {
        for _ in 0..<numOutputs {
            let value: Float32 = gaussianDist(mean: 0.0, std: std)
            //print("random: \(value)")
            result.append(value)
        }
    } else {
        for _ in 0..<numInputs*numOutputs {
            let value: Float32 = gaussianDist(mean: 0.0, std: std)
            //print("random: \(value)")
            result.append(value)
        }
    }
    
    return result
}

class Layer {
    init(numInputs: Int, numOutputs: Int) {
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        var weights = glorotInit(numInputs: numInputs, numOutputs: numOutputs, isBias: false)
        var bias = glorotInit(numInputs: numInputs, numOutputs: numOutputs, isBias: true)
        weightData = Data(bytes: &weights, count: numInputs*numOutputs*MemoryLayout<Float>.size)
        biasData = Data(bytes: &bias, count: numOutputs*MemoryLayout<Float>.size)
//        var weightsBufferShared = device.makeBuffer(bytes: &weights, length: numInputs*numOutputs*MemoryLayout<Float>.size, options: MTLResourceOptions.storageModeShared)!
//        weightsBuffer = device.makeBuffer(length: numInputs*numOutputs*MemoryLayout<Float>.size, options: MTLResourceOptions.storageModePrivate)!
//        var biasBufferShared = device.makeBuffer(bytes: &bias, length: numOutputs*MemoryLayout<Float>.size, options: MTLResourceOptions.storageModeShared)!
//        biasBuffer = device.makeBuffer(length: numOutputs*MemoryLayout<Float>.size, options: MTLResourceOptions.storageModePrivate)!
//        commandEncoder.copy(from: weightsBufferShared, sourceOffset: 0, to: weightsBuffer, destinationOffset: 0, size: numInputs*numOutputs*MemoryLayout<Float>.size)
//        commandEncoder.copy(from: biasBufferShared, sourceOffset: 0, to: biasBuffer, destinationOffset: 0, size: numOutputs*MemoryLayout<Float>.size)
    }
    var numInputs: Int
    var numOutputs: Int
//    var weightsBuffer: MTLBuffer
//    var biasBuffer: MTLBuffer
    var weightData: Data
    var biasData: Data
    //todo: later make function for copying weights and biases to memory and save
    //to JSON
}
