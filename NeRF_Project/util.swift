//
//  util.swift
//  mpsgraph_test
//
//  Created by Josh Lakin on 3/14/24.
//

import Foundation
import UniformTypeIdentifiers
import CoreGraphics
import ImageIO
import simd

func convertByteImageToFloat(image: CGImage) -> [Float] {
    print("Bits per pixel: \(image.bitsPerPixel)")
    var res: [Float] = Array(repeating: 0.0, count: 3*image.width*image.height)
    let ptr = CFDataGetBytePtr(image.dataProvider?.data)
    if (image.bitsPerPixel == 32) { //image contains alpha channel, need to get rid of it
        if (ptr != nil) {
            var byteOffset = 0
            var componentOffset = 0
            while (byteOffset < 4*image.width*image.height) {
                res[componentOffset] = Float(ptr![byteOffset]) / 255.0
                res[componentOffset + 1] = Float(ptr![byteOffset + 1]) / 255.0
                res[componentOffset + 2] = Float(ptr![byteOffset + 2]) / 255.0
                byteOffset += 4
                componentOffset += 3
            }
        }
    } else {
        var byteOffset = 0
        while (byteOffset < 3*image.width*image.height) {
            res[byteOffset] = Float(ptr![byteOffset]) / 255.0
            res[byteOffset + 1] = Float(ptr![byteOffset + 1]) / 255.0
            res[byteOffset + 2] = Float(ptr![byteOffset + 2]) / 255.0
            byteOffset += 3
        }
    }
    
    return res
}

//converts a flattened array of vector data like RGB in [Float] to an array of arrays representing vector data [[Float]]
func convertArray1DTo2D(arr: [Float], stride: Int, count: Int) -> [[Float]] {
    var result: [[Float]] = []
    for index in 0..<count {
        var elem: [Float] = []
        for comp in 0..<stride {
            elem.append(arr[index*stride+comp])
        }
        result.append(elem)
    }
    assert(result.count == count)
    return result
}

func checkArrayEqual(arr: [Float], stride: Int, count: Int, arr2d: [[Float]]) {
    for index in 0..<count {
        for comp in 0..<stride {
            let p1 = arr[index*stride+comp]
            let p2 = arr2d[index][comp]
            assert(p1 == p2)
        }
    }
}

func convertFloatImageToByte(floatImage: [Float], originalImage: CGImage, imageWidth: Int, imageHeight: Int) -> CGImage? {
    var byteImage: [UInt8] = Array(repeating: 0, count: (originalImage.bitsPerPixel / 8)*imageWidth*imageHeight)
    if (originalImage.bitsPerPixel == 32) {
        //if contains alpha channel, add alpha channel back
        var byteOffset = 0
        var componentOffset = 0
        while (componentOffset < floatImage.count) {
            byteImage[byteOffset] = UInt8(min(round(floatImage[componentOffset] * 255.0), 255.0))
            byteImage[byteOffset+1] = UInt8(min(round(floatImage[componentOffset+1] * 255.0), 255.0))
            byteImage[byteOffset+2] = UInt8(min(round(floatImage[componentOffset+2] * 255.0), 255.0))
            byteImage[byteOffset+3] = 255
            componentOffset += 3
            byteOffset += 4
        }
    } else {
        var componentOffset = 0
        while (componentOffset < floatImage.count) {
            byteImage[componentOffset] = UInt8(min(round(floatImage[componentOffset] * 255.0), 255.0))
            byteImage[componentOffset+1] = UInt8(min(round(floatImage[componentOffset+1] * 255.0), 255.0))
            byteImage[componentOffset+2] = UInt8(min(round(floatImage[componentOffset+2] * 255.0), 255.0))
            componentOffset += 3
        }
    }
    
    
    let dataOut = CFDataCreate(nil, &byteImage, byteImage.count)
    let dataProviderOut = CGDataProvider(data: dataOut!)
    let outImage = CGImage(width: originalImage.width, height: originalImage.height, bitsPerComponent: originalImage.bitsPerComponent, bitsPerPixel: originalImage.bitsPerPixel, bytesPerRow: originalImage.bytesPerRow, space: originalImage.colorSpace!, bitmapInfo: originalImage.bitmapInfo, provider: dataProviderOut!, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    return outImage
}

func frequencyEncoding(input: [Float]) -> [Float] {
    var result: [Float] = []
    for i in 0..<5 {
        result.append(sin(pow(2.0, Float(i)) * Float.pi * input[0]))
        result.append(cos(pow(2.0, Float(i)) * Float.pi * input[0]))
    }
    for i in 0..<5 {
        result.append(sin(pow(2.0, Float(i)) * Float.pi * input[1]))
        result.append(cos(pow(2.0, Float(i)) * Float.pi * input[1]))
    }
    for i in 0..<6 {
        result.append(sin(pow(2.0, Float(i)) * Float.pi * input[2]))
        result.append(cos(pow(2.0, Float(i)) * Float.pi * input[2]))
    }
    return result
}

func createInputLabels32(originalImage: CGImage) -> [[Float]] {
    var result: [[Float]] = []
    for r in 0..<originalImage.height {
        for c in 0..<originalImage.width {
            var elem: [Float] = []
            let x: Float = Float(c) / Float(originalImage.width)
            let y: Float = Float(r) / Float(originalImage.height)
            for i in 0..<8 {
                elem.append(sin(pow(2.0, Float(i)) * Float.pi * x))
                elem.append(cos(pow(2.0, Float(i)) * Float.pi * x))
            }
            for i in 0..<8 {
                elem.append(sin(pow(2.0, Float(i)) * Float.pi * y))
                elem.append(cos(pow(2.0, Float(i)) * Float.pi * y))
            }
            result.append(elem)
        }
    }
    return result
}

func createInputLabels(imageWidth: Int, imageHeight: Int) -> [[Float]] {
    var result: [[Float]] = []
    for r in 0..<imageHeight {
        for c in 0..<imageWidth {
            var arr: [Float] = []
            let x: Float = Float(c) / Float(imageWidth)
            let y: Float = Float(r) / Float(imageHeight)
            arr.append(x)
            arr.append(y)
            result.append(arr)
        }
    }
    assert(result.count == imageWidth*imageHeight, "InputCount: \(result.count) OriginalCount: \(imageWidth*imageHeight)")
    return result
}

func create2XInput(originalImage: CGImage) -> [[Float]] {
    var result: [[Float]] = []
    
    var yOffset: Float = -0.25
    for _ in 0..<2*originalImage.height {
        for c in 0..<originalImage.width {
            var arr1: [Float] = []
            var arr2: [Float] = []
            let x2: Float = (Float(c) + 0.25) / Float(originalImage.width)
            let y2: Float = (yOffset) / Float(originalImage.height)
            let x1: Float = (Float(c) - 0.25) / Float(originalImage.width)
            let y1: Float = (yOffset) / Float(originalImage.height)
            arr1.append(x1)
            arr1.append(y1)
            arr2.append(x2)
            arr2.append(y2)
            result.append(arr1)
            result.append(arr2)
        }
        yOffset += 0.5
    }
    assert(result.count == 2*2*originalImage.width*originalImage.height, "InputCount: \(result.count) OriginalCount: \(originalImage.width*originalImage.width)")
    return result
}

func createInputLabels3D(originalImage: CGImage) -> [[Float]] {
    var result: [[Float]] = []
    for r in 0..<originalImage.height {
        for c in 0..<originalImage.width {
            var arr: [Float] = []
            let x: Float = Float(c) / Float(originalImage.width)
            let y: Float = Float(r) / Float(originalImage.height)
            arr.append(x)
            arr.append(y)
            arr.append(-1)
            result.append(arr)
        }
    }
    assert(result.count == originalImage.width*originalImage.height, "InputCount: \(result.count) OriginalCount: \(originalImage.width*originalImage.height)")
    return result
}

func rotateVector(dir: simd_float3, axis: simd_float3, angle: Float) -> simd_float3 {
    let k_cross_v = simd_cross(axis, dir)
    let sinAngle = sinf(angle)
    let oneMinusCosAngle = 1.0 - cosf(angle)
    let scaledK = oneMinusCosAngle*axis
    let temp = simd_cross(scaledK, k_cross_v)
    let result = dir + temp + sinAngle*k_cross_v
    return result
}

func castRay(cameraDir: simd_float3, cameraPos: simd_float3, numSteps: Int, minWorldBound: simd_float3, maxWorldBound: simd_float3, deltaValues: [Float]) -> [Float] {
    var result: [Float] = []
    var currentT: Float = 1.0
    for i in 0..<numSteps {
        var rayPos = cameraPos + currentT*cameraDir
        currentT = currentT + deltaValues[i]
        //var rayPosClipped = max(rayPos, minWorldBound)
        //rayPosClipped = min(rayPosClipped, maxWorldBound)
        let worldBoundLength = maxWorldBound - minWorldBound
        rayPos = (rayPos - minWorldBound) / worldBoundLength
        result.append(rayPos.x)
        result.append(rayPos.y)
        result.append(rayPos.z)
    }
    return result
}

func castRayPosEncoding(cameraDir: simd_float3, cameraPos: simd_float3, numSteps: Int, minWorldBound: simd_float3, maxWorldBound: simd_float3, deltaValues: [Float]) -> [Float] {
    var result: [Float] = []
    var currentT: Float = 1.0
    for i in 0..<numSteps {
        var rayPos = cameraPos + currentT*cameraDir
        currentT = currentT + deltaValues[i]
        //var rayPosClipped = max(rayPos, minWorldBound)
        //rayPosClipped = min(rayPosClipped, maxWorldBound)
        let worldBoundLength = maxWorldBound - minWorldBound
        rayPos = (rayPos - minWorldBound) / worldBoundLength
        let posEncoding = frequencyEncoding(input: [rayPos.x, rayPos.y, rayPos.z])
        result.append(contentsOf: posEncoding)
    }
    return result
}

func createNeRFInputData(originalImage: CGImage, rowStart: Int, colStart: Int, blockWidth: Int, blockHeight: Int, numSteps: Int, maxT: Double, cameraFocalLength: Float, cameraPos: simd_float3, cameraAxis: simd_float3, cameraAngle: Float, minWorldBound: simd_float3, maxWorldBound: simd_float3) -> ([[Float]], [[Float]], [Float]) {
    var inputValues: [[Float]] = []
    var dirValues: [[Float]] = []
    var deltaValues: [Float] = []
    var currentT: Float = 1.0
    for i in 0..<numSteps {
        let b = exp( (log(maxT) - log(1.0)) / Double(numSteps))
        let newT = Float(1.0*pow(b, Double(i+1)))
        var deltaT = newT-currentT
        deltaValues.append(deltaT)
        currentT = newT
        print("t[\(i)]: \(currentT), deltaT[\(i)]: \(deltaT)")
    }
    for r in rowStart..<blockHeight {
        for c in colStart..<blockWidth {
            autoreleasepool {
                var cameraDirArr: [Float] = []
                var x: Float = Float(c) / Float(originalImage.width)
                x = x - 0.5
                var y: Float = Float(r) / Float(originalImage.height)
                y = y - 0.5;
                let cameraDir = simd_float3(x: x, y: y, z: -cameraFocalLength)
                let cameraDirRotated = rotateVector(dir: cameraDir, axis: cameraAxis, angle: cameraAngle)
                let raySteps = castRay(cameraDir: cameraDirRotated, cameraPos: cameraPos, numSteps: numSteps, minWorldBound: minWorldBound, maxWorldBound: maxWorldBound, deltaValues: deltaValues)
                inputValues.append(raySteps)
                let cameraDirNormalized = simd_normalize(cameraDirRotated)
                cameraDirArr.append(cameraDirNormalized.x)
                cameraDirArr.append(cameraDirNormalized.y)
                cameraDirArr.append(cameraDirNormalized.z)
                dirValues.append(cameraDirArr)
            }
            
        }
    }
    return (inputValues, dirValues, deltaValues)
}

func createBatchesNeRFBad(imageWidth: Int, imageHeight: Int, batchSize: Int, floatImage: [[Float]], numSteps: Int, cameraPos: simd_float3, cameraAxis: simd_float3, cameraAngle: Float, minWorldBound: simd_float3, maxWorldBound: simd_float3) -> ([[Float]], [[Float]], [Float], [[Float]]) {
    var inputResult: [[Float]] = []
    var dirBatches: [[Float]] = []
    var deltaValues: [Float] = []
    var currentT: Float = 1.0
    for i in 0..<numSteps {
        let b = exp( (log(5.0) - log(1.0)) / Double(numSteps))
        let newT = Float(1.0*pow(b, Double(i+1)))
        let deltaT = newT-currentT
        deltaValues.append(deltaT)
        currentT = newT
        print("t \(i): \(currentT), deltaT \(i): \(deltaT)")
    }
    var labelResult: [[Float]] = []
    let batchGridLength: Int = min( Int(sqrt(Float(batchSize))), imageWidth, imageHeight)
    for r_start in stride(from: 0, to: imageHeight, by: batchGridLength) {
        for c_start in stride(from: 0, to: imageWidth, by: batchGridLength) {
            let r_size = min(imageHeight-r_start, batchGridLength)
            let c_size = min(imageWidth-c_start, batchGridLength)
            var inputArr: [Float] = []
            var labelArr: [Float] = []
            var dirArr: [Float] = []
            for r in 0..<r_size {
                for c in 0..<c_size {
                    let imageIndex = imageWidth*(r + r_start) + (c + c_start)
                    labelArr.append(floatImage[imageIndex][0])
                    labelArr.append(floatImage[imageIndex][1])
                    labelArr.append(floatImage[imageIndex][2])
                    var x: Float = Float(c_start+c) - (Float(imageWidth) / 2.0)
                    //x /= Float(imageWidth)
                    var y: Float = -(Float(r_start+r) - (Float(imageHeight) / 2.0))
                    //y /= Float(imageHeight)
                    let z: Float = -1
                    var dir = simd_float3(x: x, y: y, z: z)
                    dir = rotateVector(dir: dir, axis: cameraAxis, angle: cameraAngle)
                    let normalizedDir = simd_normalize(dir)
                    dirArr.append(normalizedDir.x)
                    dirArr.append(normalizedDir.y)
                    dirArr.append(normalizedDir.z)
                    let raySteps = castRay(cameraDir: dir, cameraPos: cameraPos, numSteps: numSteps, minWorldBound: minWorldBound, maxWorldBound: maxWorldBound, deltaValues: deltaValues)
                    inputArr.append(contentsOf: raySteps)
                }
            }
            inputResult.append(inputArr)
            dirBatches.append(dirArr)
            labelResult.append(labelArr)
        }
    }
    assert(inputResult.count == labelResult.count)
    return (inputResult, dirBatches, deltaValues, labelResult)
}

class CodableWeights : Codable {
    var weightMatrices: [[Float]] = []
    var biasVectors: [[Float]] = []
    var hashTableWeights: [[Float]] = []
    var cameraPos: [[Float]] = []
    var cameraMatrix: [[Float]] = []
}

func convertDepthImageToRGB(depthImage: [Float], numSteps: Int) -> [Float] {
    autoreleasepool {
        var result: [Float] = []
        for x in depthImage {
            var g = x / Float(numSteps)
            //g = 1.0 - g
            result.append(g)
            result.append(g)
            result.append(g)
        }
        return result
    }
}

func convertDepthImageToRGB2(depthImage: [Float]) -> [Float] {
    autoreleasepool {
        var result: [Float] = []
        var maxTrans: Float = 0.0
        for x in depthImage {
            if (x > maxTrans) {
                maxTrans = x
            }
        }
        for x in depthImage {
            var g = x / (max(0.001, maxTrans))
            g = 1.0 - g
            result.append(g)
            result.append(g)
            result.append(g)
        }
        return result
    }
}

func generateCameraPosAngleCircleImage(imageIndex: Int, numImages: Int, radius: Float, center: simd_float3) -> (Float, simd_float3) {
    let two_pi = 2.0*Float.pi
    let ratio = Float(imageIndex - 1) / Float(numImages)
    let angle = ratio*two_pi
    var cameraPos = simd_float3(x: sinf(angle), y: 0.0, z: -cosf(angle))
    cameraPos = center + radius*cameraPos
    return (angle, cameraPos)
}

func generateCameraPosAngleCircle(scale: Float, radius: Float, center: simd_float3) -> (Float, simd_float3) {
    let two_pi = 2.0*Float.pi
    let angle = scale*two_pi
    var cameraPos = simd_float3(x: sinf(angle), y: 0.0, z: -cosf(angle))
    cameraPos = center + radius*cameraPos
    return (angle, cameraPos)
}

func zipInputsLabels(inputs: [[Float]], labels: [[Float]]) -> [([Float], [Float])] {
    var result: [([Float], [Float])] = []
    assert(inputs.count == labels.count)
    for i in 0..<inputs.count {
        let pair: ([Float], [Float]) = (inputs[i], labels[i])
        result.append(pair)
    }
    return result
}

func zipInputsLabelsImageIndex(inputs: [[Float]], labels: [[Float]], imageIndex: Int) ->[([Float], [Float], Int)] {
    var result: [([Float], [Float], Int)] = []
    assert(inputs.count == labels.count)
    for i in 0..<inputs.count {
        let triplet: ([Float], [Float], Int) = (inputs[i], labels[i], imageIndex)
        result.append(triplet)
    }
    return result
}
