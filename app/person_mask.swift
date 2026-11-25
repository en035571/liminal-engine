// Person cut-out for the Granular tab's Light bodies look (11-24): Apple Vision person segmentation, on-device.
// Protocol (stdin/stdout, binary): argv = width height quality(0 fast, 1 balanced, 2 accurate)
//   in : per frame, width*height*4 bytes RGBA
//   out: per frame, int32 mask width, int32 mask height (little-endian), then mw*mh bytes (0 = background, 255 = person)
// One VNSequenceRequestHandler for the whole clip, so Vision can keep the mask steady from frame to frame.
// Build: swiftc -O person_mask.swift -o person_mask   (granular.py builds it on first use)
import Foundation
import Vision
import CoreVideo

let args = CommandLine.arguments
guard args.count >= 3, let W = Int(args[1]), let H = Int(args[2]) else {
    FileHandle.standardError.write("usage: person_mask width height [quality]\n".data(using: .utf8)!); exit(2)
}
let q = args.count > 3 ? Int(args[3]) ?? 1 : 1
let req = VNGeneratePersonSegmentationRequest()
req.qualityLevel = q == 0 ? .fast : (q == 2 ? .accurate : .balanced)
req.outputPixelFormat = kCVPixelFormatType_OneComponent8
let seq = VNSequenceRequestHandler()
let inH = FileHandle.standardInput, outH = FileHandle.standardOutput
let n = W * H * 4

func readExactly(_ count: Int) -> Data? {
    var d = Data(capacity: count)
    while d.count < count {
        let chunk = inH.readData(ofLength: count - d.count)
        if chunk.isEmpty { return nil }
        d.append(chunk)
    }
    return d
}

while let frame = readExactly(n) {
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(nil, W, H, kCVPixelFormatType_32BGRA, nil, &pb)
    guard let buf = pb else { exit(3) }
    CVPixelBufferLockBaseAddress(buf, [])
    let base = CVPixelBufferGetBaseAddress(buf)!.assumingMemoryBound(to: UInt8.self)
    let bpr = CVPixelBufferGetBytesPerRow(buf)
    frame.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
        let s = src.bindMemory(to: UInt8.self)
        for y in 0..<H {
            let row = base + y * bpr
            for x in 0..<W {
                let i = (y * W + x) * 4
                row[x * 4] = s[i + 2]; row[x * 4 + 1] = s[i + 1]; row[x * 4 + 2] = s[i]; row[x * 4 + 3] = 255   // RGBA -> BGRA
            }
        }
    }
    CVPixelBufferUnlockBaseAddress(buf, [])
    var mw: Int32 = 0, mh: Int32 = 0
    var bytes = Data()
    do {
        try seq.perform([req], on: buf)
        if let m = req.results?.first?.pixelBuffer {
            CVPixelBufferLockBaseAddress(m, .readOnly)
            mw = Int32(CVPixelBufferGetWidth(m)); mh = Int32(CVPixelBufferGetHeight(m))
            let mb = CVPixelBufferGetBaseAddress(m)!.assumingMemoryBound(to: UInt8.self)
            let mbpr = CVPixelBufferGetBytesPerRow(m)
            bytes.reserveCapacity(Int(mw * mh))
            for y in 0..<Int(mh) { bytes.append(mb + y * mbpr, count: Int(mw)) }
            CVPixelBufferUnlockBaseAddress(m, .readOnly)
        }
    } catch {
        FileHandle.standardError.write("vision error: \(error)\n".data(using: .utf8)!)
    }
    var hdr = Data()
    withUnsafeBytes(of: mw.littleEndian) { hdr.append(contentsOf: $0) }
    withUnsafeBytes(of: mh.littleEndian) { hdr.append(contentsOf: $0) }
    outH.write(hdr); if mw > 0 { outH.write(bytes) }
}
