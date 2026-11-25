import Accelerate
import Foundation

/// An on-device dense text embedder behind a narrow protocol, matching
/// `AnswerModel`'s boundary so `AskIndexStore`, `SemanticRetriever` and
/// `AskEngine` stay independent of CoreML.
public protocol EmbeddingModel: Sendable {
    /// Stable identifier stored beside each vector in `embed_span.model` so
    /// switching embedding models re-embeds cleanly without mixing vector spaces.
    var modelID: String { get }

    /// Embeds a batch of strings, returning one L2-normalised vector per input.
    func embed(_ texts: [String], mode: EmbeddingMode) async throws -> [[Float]]
}

/// Asymmetric query vs. document prompt mode (matches EmbeddingGemma-2's
/// trained prefixes: retrieval quality drops if queries and passages share one).
public enum EmbeddingMode: String, Sendable, Hashable, Codable {
    case query
    case document
}

/// Vector math, compact SQLite serialisation, and cross-architecture IEEE-754
/// half-precision conversion backed by Apple Accelerate (`vDSP`).
public enum EmbeddingMath {
    /// L2-normalises `vector` (returns unchanged if norm is near zero).
    public static func l2Normalize(_ vector: [Float]) -> [Float] {
        guard !vector.isEmpty else { return vector }
        var sumSq: Float = 0
        vDSP_svesq(vector, 1, &sumSq, vDSP_Length(vector.count))
        let norm = sqrtf(sumSq)
        guard norm > 1e-12 else { return vector }
        var inverse = 1.0 / norm
        var out = [Float](repeating: 0, count: vector.count)
        vDSP_vsmul(vector, 1, &inverse, &out, 1, vDSP_Length(vector.count))
        return out
    }

    /// Dot product of two L2-normalised vectors (= cosine similarity).
    /// When `scoreDim` is set and smaller than either vector, slices to the
    /// first `scoreDim` components and L2-renormalises for Matryoshka scoring.
    public static func cosine(_ first: [Float], _ second: [Float], scoreDim: Int? = nil) -> Float {
        let count = min(first.count, second.count, scoreDim ?? Int.max)
        guard count > 0 else { return 0 }
        var dot: Float = 0
        first.withUnsafeBufferPointer { firstBuffer in
            second.withUnsafeBufferPointer { secondBuffer in
                guard let a = firstBuffer.baseAddress, let b = secondBuffer.baseAddress else { return }
                vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(count))
            }
        }
        if let dim = scoreDim, dim < first.count || dim < second.count {
            var normFirst: Float = 0
            var normSecond: Float = 0
            first.withUnsafeBufferPointer { buffer in
                if let base = buffer.baseAddress { vDSP_svesq(base, 1, &normFirst, vDSP_Length(count)) }
            }
            second.withUnsafeBufferPointer { buffer in
                if let base = buffer.baseAddress { vDSP_svesq(base, 1, &normSecond, vDSP_Length(count)) }
            }
            let denominator = sqrtf(normFirst) * sqrtf(normSecond)
            return denominator > 1e-12 ? dot / denominator : 0
        }
        return dot
    }

    /// Packs a `[Float]` as little-endian IEEE-754 float32 bytes for SQLite `BLOB`.
    public static func encodeFloat32(_ vector: [Float]) -> Data {
        var data = Data(capacity: vector.count * MemoryLayout<UInt32>.size)
        for value in vector {
            var littleEndian = value.bitPattern.littleEndian
            withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }

    /// Unpacks little-endian float32 `BLOB` bytes back into `[Float]` safely
    /// without assuming pointer alignment on sliced `Data`.
    public static func decodeFloat32(_ data: Data) -> [Float] {
        let stride = MemoryLayout<UInt32>.size
        let count = data.count / stride
        guard count > 0 else { return [] }
        return data.withUnsafeBytes { raw in
            var out = [Float](repeating: 0, count: count)
            for index in 0..<count {
                let bits = raw.loadUnaligned(fromByteOffset: index * stride, as: UInt32.self)
                out[index] = Float(bitPattern: UInt32(littleEndian: bits))
            }
            return out
        }
    }

    /// IEEE-754 binary16 (`Float16`) conversion that compiles on every Apple
    /// architecture, including universal macOS builds (`x86_64` lacks `Float16`).
    public enum HalfFloat {
        /// IEEE-754 binary16 bit pattern for `1.0`.
        public static let oneBitPattern: UInt16 = 0x3C00

        @inline(__always)
        public static func toFloat(_ bits: UInt16) -> Float {
            #if !arch(x86_64)
            return Float(Float16(bitPattern: bits))
            #else
            return decodeIEEE754Half(bits)
            #endif
        }

        @inline(__always)
        public static func fromFloat(_ value: Float) -> UInt16 {
            #if !arch(x86_64)
            return Float16(value).bitPattern
            #else
            return encodeIEEE754Half(value)
            #endif
        }

        /// Software IEEE-754 binary16 → binary32 conversion (also exposed for
        /// cross-architecture unit testing on arm64).
        static func decodeIEEE754Half(_ bits: UInt16) -> Float {
            let sign = UInt32(bits & 0x8000) << 16
            let exponent = Int((bits >> 10) & 0x1F)
            let mantissa = UInt32(bits & 0x03FF)

            if exponent == 0 {
                if mantissa == 0 {
                    return Float(bitPattern: sign)
                }
                // Subnormal: normalise mantissa.
                var shift = 0
                var m = mantissa
                while (m & 0x0400) == 0 {
                    m <<= 1
                    shift += 1
                }
                let exp32 = UInt32(127 - 14 - shift) << 23
                let mant32 = (m & 0x03FF) << 13
                return Float(bitPattern: sign | exp32 | mant32)
            }
            if exponent == 31 {
                let mant32 = mantissa << 13
                return Float(bitPattern: sign | 0x7F80_0000 | mant32)
            }
            let exp32 = UInt32(exponent + (127 - 15)) << 23
            let mant32 = mantissa << 13
            return Float(bitPattern: sign | exp32 | mant32)
        }

        /// Software IEEE-754 binary32 → binary16 round-to-nearest-even conversion
        /// (exposed so arm64 unit tests verify parity against hardware `Float16`).
        static func encodeIEEE754Half(_ value: Float) -> UInt16 {
            let bits = value.bitPattern
            let sign = UInt16((bits >> 16) & 0x8000)
            let exponent = Int((bits >> 23) & 0xFF) - 127
            let mantissa = bits & 0x007F_FFFF

            if exponent == 128 {
                // Inf or NaN.
                if mantissa == 0 { return sign | 0x7C00 }
                return sign | 0x7E00 | UInt16(mantissa >> 13)
            }
            if exponent > 15 {
                // Overflow to infinity.
                return sign | 0x7C00
            }
            if exponent < -14 {
                if exponent < -24 {
                    return sign
                }
                // Subnormal half.
                let fullMantissa = mantissa | 0x0080_0000
                let shift = UInt32(-14 - exponent + 13)
                let halfMantissa = fullMantissa >> shift
                let remainder = fullMantissa & ((1 << shift) - 1)
                let halfway: UInt32 = 1 << (shift - 1)
                let rounded = halfMantissa + ((remainder > halfway || (remainder == halfway && (halfMantissa & 1) == 1)) ? 1 : 0)
                return sign | UInt16(rounded)
            }

            let biased = UInt16(exponent + 15) << 10
            let halfMantissa = UInt16(mantissa >> 13)
            let remainder = mantissa & 0x1FFF
            let roundUp: UInt16 = (remainder > 0x1000 || (remainder == 0x1000 && (halfMantissa & 1) == 1)) ? 1 : 0
            return sign | (biased + halfMantissa + roundUp)
        }
    }
}
