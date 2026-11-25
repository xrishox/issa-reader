import CryptoKit
import Foundation
import Synchronization
import Testing

@testable import IssaAsk

private final class MockDownloadURLProtocol: URLProtocol, @unchecked Sendable {
    struct Response: Sendable {
        let statusCode: Int
        let body: Data
    }

    private static let routes = Mutex<[String: Response]>([:])

    static func register(_ map: [String: Response]) {
        routes.withLock { $0 = map }
    }

    private static func response(for url: URL) -> Response? {
        routes.withLock { $0[url.absoluteString] }
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url, let route = Self.response(for: url) else {
            client?.urlProtocol(self, didFailWithError: URLError(.fileDoesNotExist))
            return
        }
        let http = HTTPURLResponse(url: url, statusCode: route.statusCode, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: http, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: route.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

@Suite("CoreMLEmbeddingModelTests", .serialized)
struct CoreMLEmbeddingModelTests {
    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    @Test("decodes unaligned Data slices without assuming pointer alignment")
    func unalignedFloat32Decode() {
        let original: [Float] = [0.25, -1.5, 3.14159, 0.0, 42.0]
        let encoded = EmbeddingMath.encodeFloat32(original)
        var padded = Data([0xAA])
        padded.append(encoded)
        let slice = padded.subdata(in: 1..<padded.count)
        #expect(EmbeddingMath.decodeFloat32(slice) == original)
    }

    @Test("software IEEE-754 half-precision conversion matches hardware Float16")
    func halfFloatParity() {
        #if !arch(x86_64)
        let samples: [Float] = [0.0, -0.0, 1.0, -1.0, 0.5, 22.627417, 1e-5, -65504.0, 65504.0, 1e10]
        for value in samples {
            let hwBits = Float16(value).bitPattern
            let swBits = EmbeddingMath.HalfFloat.encodeIEEE754Half(value)
            #expect(swBits == hwBits)
            let hwBack = Float(Float16(bitPattern: hwBits))
            let swBack = EmbeddingMath.HalfFloat.decodeIEEE754Half(hwBits)
            #expect(swBack.bitPattern == hwBack.bitPattern)
        }
        #endif
    }

    #if canImport(CoreML) && !os(tvOS)
    @Test("routes all sequence lengths to the single resident embed_512 bucket")
    func bucketSelection() {
        #expect(CoreMLEmbeddingModel.bucket(forLength: 1) == 512)
        #expect(CoreMLEmbeddingModel.bucket(forLength: 32) == 512)
        #expect(CoreMLEmbeddingModel.bucket(forLength: 128) == 512)
        #expect(CoreMLEmbeddingModel.bucket(forLength: 400) == 512)
        #expect(CoreMLEmbeddingModel.bucket(forLength: 600) == 512)
    }

    @Test("empty model directory reports notInstalled, prewarm is a safe no-op, and embed refuses")
    func missingDirectoryRefusesEmbed() async throws {
        let dir = try AskFixture.temporaryDirectory()
        defer { AskFixture.remove(dir) }
        let model = CoreMLEmbeddingModel(modelDirectory: dir)
        #expect(!model.isInstalledOnDisk())
        #expect(!(await model.isModelResident()))
        #expect(!(await model.prewarm()))
        await #expect(throws: CoreMLEmbeddingModel.ModelError.modelNotInstalled) {
            _ = try await model.embed(["hello"], mode: .query)
        }
    }

    @Test("vector validation rejects empty, zero-norm, and non-finite vectors")
    func vectorValidation() {
        #expect(!CoreMLEmbeddingModel.isValidVector([]))
        #expect(!CoreMLEmbeddingModel.isValidVector([0, 0, 0]))
        #expect(!CoreMLEmbeddingModel.isValidVector([1, .nan, 0]))
        #expect(!CoreMLEmbeddingModel.isValidVector([1, .infinity, 0]))
        #expect(CoreMLEmbeddingModel.isValidVector([0.6, 0.8]))
    }

    @Test("downloadModel verifies SHA-256, installs atomically, sweeps stale staging, and rejects corruption")
    func downloadVerificationAndFailureModes() async throws {
        let root = try AskFixture.temporaryDirectory()
        defer { AskFixture.remove(root) }
        let modelDir = root.appending(path: "test-model", directoryHint: .isDirectory)

        // Seed an orphaned staging folder to prove `downloadModel` sweeps it.
        let orphan = root.appending(path: "test-model.downloading-orphan", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: orphan, withIntermediateDirectories: true)

        let payloadA = Data("config-bytes".utf8)
        let payloadB = Data(repeating: 0x42, count: 4_096)
        let baseURL = URL(string: "https://example.invalid/model/")!
        let manifest = [
            CoreMLEmbeddingModel.ManifestFile(
                relativePath: "config.json",
                sizeBytes: Int64(payloadA.count),
                sha256: Self.sha256Hex(payloadA),
            ),
            CoreMLEmbeddingModel.ManifestFile(
                relativePath: "text/weights.bin",
                sizeBytes: Int64(payloadB.count),
                sha256: Self.sha256Hex(payloadB),
            ),
        ]

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockDownloadURLProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }

        let model = CoreMLEmbeddingModel(
            modelDirectory: modelDir,
            modelID: "test-model",
            baseURL: baseURL,
            manifest: manifest,
            session: session,
        )

        // 1. Checksum mismatch must fail and leave `isInstalledOnDisk() == false`.
        MockDownloadURLProtocol.register([
            baseURL.appending(path: "config.json").absoluteString: .init(statusCode: 200, body: Data("tampered!!!!".utf8)),
            baseURL.appending(path: "text/weights.bin").absoluteString: .init(statusCode: 200, body: payloadB),
        ])
        await #expect(throws: CoreMLEmbeddingModel.ModelError.checksumMismatch) {
            try await model.downloadModel()
        }
        #expect(!model.isInstalledOnDisk())
        #expect(!FileManager.default.fileExists(atPath: orphan.path(percentEncoded: false)))

        // 2. HTTP 500 must fail with `.downloadFailed`.
        MockDownloadURLProtocol.register([
            baseURL.appending(path: "config.json").absoluteString: .init(statusCode: 500, body: Data()),
        ])
        await #expect(throws: CoreMLEmbeddingModel.ModelError.downloadFailed(statusCode: 500)) {
            try await model.downloadModel()
        }
        #expect(!model.isInstalledOnDisk())

        // 3. Valid download installs atomically and reports `isInstalledOnDisk() == true`.
        MockDownloadURLProtocol.register([
            baseURL.appending(path: "config.json").absoluteString: .init(statusCode: 200, body: payloadA),
            baseURL.appending(path: "text/weights.bin").absoluteString: .init(statusCode: 200, body: payloadB),
        ])
        try await model.downloadModel()
        #expect(model.isInstalledOnDisk())

        // 4. `removeModel()` deletes files cleanly.
        await model.removeModel()
        #expect(!model.isInstalledOnDisk())
    }

    @Test(
        "live CoreML Neural Engine download and inference",
        .enabled(if: ProcessInfo.processInfo.environment["ISSA_COREML_LIVE_TEST"] == "1"),
    )
    func liveANEInference() async throws {
        let dir = ProcessInfo.processInfo.environment["ISSA_COREML_MODEL_DIR"].map(URL.init(fileURLWithPath:))
            ?? FileManager.default.temporaryDirectory.appending(path: "issa-eg2-coreml-live", directoryHint: .isDirectory)
        let model = CoreMLEmbeddingModel(modelDirectory: dir)
        if !model.isInstalledOnDisk() {
            try await model.downloadModel()
        }
        #expect(model.isInstalledOnDisk())

        let queryVectors = try await model.embed(
            ["Where was Mollie hiding during the battle?"],
            mode: .query,
        )
        let docVectors = try await model.embed(
            [
                "Mollie was missing after the Battle of the Cowshed, and she was found hiding in her stall with her head buried among the hay in the manger.",
                "Snowball studied an old book of Julius Caesar's campaigns which he had found in the farmhouse and planned the defensive operations.",
            ],
            mode: .document,
        )
        #expect(queryVectors.count == 1)
        #expect(docVectors.count == 2)
        #expect(queryVectors[0].count == 768)

        let relevant = EmbeddingMath.cosine(queryVectors[0], docVectors[0])
        let unrelated = EmbeddingMath.cosine(queryVectors[0], docVectors[1])
        #expect(relevant > unrelated + 0.1)
        #expect(relevant > 0.4)
        await model.unloadAll()
    }
    #endif
}
