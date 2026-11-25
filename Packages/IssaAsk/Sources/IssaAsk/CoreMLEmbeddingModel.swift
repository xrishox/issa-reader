#if canImport(CoreML) && !os(tvOS)
import CoreML
import CryptoKit
import Foundation
import IssaCore

/// Installation state of the optional on-device search model.
public enum EmbeddingModelState: Sendable, Equatable {
    case notInstalled
    case downloading(progress: Double)
    case ready
    case failed(String)

    public var isReady: Bool {
        if case .ready = self { return true }
        return false
    }
}

/// On-device INT8 CoreML runner for `google/embeddinggemma-2`, using the
/// precompiled Apple Neural Engine package (`OpenInfinity/embeddinggemma-2-coreml-int8`,
/// Apache 2.0 packaging; ~313 MB text-only).
///
/// Design invariants:
/// - **Pinned revision + SHA-256 verification:** Every URL points at a pinned
///   Hugging Face commit SHA (`pinnedRevision`) matching `textManifest`'s
///   SHA-256 digests, so an upstream repo edit can never silently break downloads.
/// - **Chunked disk streaming:** Uses `URLSession.download(for:delegate:)` rather
///   than byte-by-byte async iteration, hashes each file in 1 MB chunks off the
///   actor, checks free space first, and sweeps orphaned `.downloading-*` folders.
/// - **Outside `StorageRoot/Ask`:** Lives under `StorageRoot/AskModels/` so
///   signing out of an account (which purges `Ask/`) never deletes the shared
///   313 MB model. Marked excluded from iCloud backup.
/// - **Bounded iPhone RAM:** Memory-maps the 134 MB INT8 token table, caches a
///   single resident `embed_512` CoreML function (`maxLoadedBuckets = 1`),
///   loads the tokeniser and CoreML model off-actor via async `MLModel.load`,
///   and locks `MLMultiArray` buffers via `withUnsafe[Mutable]Bytes` with
///   explicit `strides` so ANE `IOSurface` padding never misreads memory.
public actor CoreMLEmbeddingModel: EmbeddingModel {
    public enum ModelError: Error, Sendable, Equatable {
        case modelNotInstalled
        case insufficientStorage
        case corruptAsset
        case checksumMismatch
        case downloadFailed(statusCode: Int)
        case inferenceFailed

        /// Plain reader-facing message — never exposes internal filenames,
        /// paths or framework error text, matching `AskFailure.message`.
        public var userFacingMessage: String {
            switch self {
            case .modelNotInstalled:
                "The on-device search model isn't downloaded yet."
            case .insufficientStorage:
                "Not enough free storage to download the search model."
            case .corruptAsset:
                "The search model files look damaged. Try downloading them again."
            case .checksumMismatch:
                "The download didn't verify cleanly. Try again."
            case .downloadFailed:
                "Couldn't download the search model right now. Check your connection and try again."
            case .inferenceFailed:
                "Couldn't run the on-device search model."
            }
        }
    }

    public struct ManifestFile: Sendable, Equatable {
        public let relativePath: String
        public let sizeBytes: Int64
        public let sha256: String

        public init(relativePath: String, sizeBytes: Int64, sha256: String) {
            self.relativePath = relativePath
            self.sizeBytes = sizeBytes
            self.sha256 = sha256
        }
    }

    public static let defaultModelID = "embeddinggemma-2-coreml-int8"
    /// Pinned Hugging Face commit SHA corresponding to `textManifest`'s digests.
    public static let pinnedRevision = "4a99ef141d6866562f64fcd1db08657bb8e9b6ba"
    public static let defaultBaseURL = URL(
        string: "https://huggingface.co/OpenInfinity/embeddinggemma-2-coreml-int8/resolve/\(pinnedRevision)/"
    )!

    /// Exact pinned manifest of the 8 text-only files from `SHA256SUMS`.
    public static let textManifest: [ManifestFile] = [
        ManifestFile(
            relativePath: "config.json",
            sizeBytes: 3_110,
            sha256: "ff37d6627c6823228eddaa4b45edf7daf694b2b9d6b45d6f0147a68c5cf2ab55",
        ),
        ManifestFile(
            relativePath: "text/EmbeddingGemma2Text.mlmodelc/analytics/coremldata.bin",
            sizeBytes: 243,
            sha256: "cbca8ceddbfe6edab5888ec5341b48fb9ad3d08117f84c36fd81837fd8c8a9bd",
        ),
        ManifestFile(
            relativePath: "text/EmbeddingGemma2Text.mlmodelc/coremldata.bin",
            sizeBytes: 972,
            sha256: "b494840b269a9cb533d7249f6fb54194e9cdd3bd318be464f01ebda87ab72d16",
        ),
        ManifestFile(
            relativePath: "text/EmbeddingGemma2Text.mlmodelc/model.mil",
            sizeBytes: 7_619_255,
            sha256: "437fb306eb2bd5c4889126a21e65622cd0423a55d77737b8a4be55f189143723",
        ),
        ManifestFile(
            relativePath: "text/EmbeddingGemma2Text.mlmodelc/weights/weight.bin",
            sizeBytes: 138_959_936,
            sha256: "60b4533a5103ebc2aab25ebcb8c0b9911f0793f7209e8f3858a033f58377a8e5",
        ),
        ManifestFile(
            relativePath: "text/embeddings.int8",
            sizeBytes: 134_217_728,
            sha256: "4b940b7da3ee169138c07d0a7383a6f3f67aaee04dad6f47d7b4cac88b74ec77",
        ),
        ManifestFile(
            relativePath: "text/embeddings.int8.scale.f16",
            sizeBytes: 524_288,
            sha256: "e77843e932248346050e51b00a6b9af1bfbc18ed75e4adb82d41f08c29218d25",
        ),
        ManifestFile(
            relativePath: "text/tokenizer.json",
            sizeBytes: 32_170_510,
            sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
        ),
    ]

    public static let totalDownloadBytes: Int64 = textManifest.reduce(0) { $0 + $1.sizeBytes }

    public enum Limits {
        static let vocabSize = 262_144
        static let hiddenSize = 512
        static let embeddingDim = 768
        /// Gemma multiplies input token embeddings by √(hidden_size) = √512 ≈ 22.627417.
        static let embedScale: Float = 22.627417
        static let maxSequenceLength = 512
        /// Pin all spans and queries to a single `embed_512` CoreML function.
        /// Measured on iPhone 16 Pro Max: each sequence-length function incurs
        /// a 38–62 s cold ANE compile and ~1.75 s per LRU eviction reload when
        /// spans alternate across buckets, whereas keeping one `embed_512`
        /// handle resident compiles once, halves RAM, and eliminates thrashing.
        static let supportedBuckets = [512]
        static let maxLoadedBuckets = 1
        /// Extra headroom required beyond the raw download size so staging +
        /// atomic replace never run the device out of storage.
        static let freeSpaceHeadroomBytes: Int64 = 128 * 1_024 * 1_024
        static let hashChunkBytes = 1_048_576
    }

    private static let documentPrefix = "title: none | text: "
    private static let queryPrefix = "task: search result | query: "
    private static let stagingPrefix = ".downloading-"

    public nonisolated let modelID: String
    public nonisolated let modelDirectory: URL
    private let baseURL: URL
    private let manifest: [ManifestFile]
    private let session: URLSession
    private let ownsSession: Bool

    /// Apple documents `MLModel` as safe for concurrent `prediction(from:)` calls;
    /// wrapped here so Swift 6 strict concurrency lets predictions suspend off
    /// this actor's executor without holding it during ANE work.
    private struct SendableMLModel: @unchecked Sendable {
        let model: MLModel
    }

    // Lazily loaded runtime state.
    private var tokenizer: GemmaTokenizer?
    private var tokenTableData: Data?
    private var rowScales: [Float]?
    private var loadedBuckets: [Int: SendableMLModel] = [:]
    private var bucketUseOrder: [Int] = []

    // Single-flight async load tasks + invalidation generations so `unloadModels()`
    // / `unloadAll()` never block behind a 20–63 s cold ANE compile and an
    // obsolete load can never repopulate caches after an unload.
    private var modelUnloadGeneration: UInt64 = 0
    private var tokenizerUnloadGeneration: UInt64 = 0
    private var tokenizerLoadTask: Task<GemmaTokenizer, any Error>?
    private var tableLoadTask: Task<(Data, [Float]), any Error>?
    private var bucketLoadTasks: [Int: Task<SendableMLModel, any Error>] = [:]

    public init(
        modelDirectory: URL? = nil,
        modelID: String = CoreMLEmbeddingModel.defaultModelID,
        baseURL: URL = CoreMLEmbeddingModel.defaultBaseURL,
        manifest: [ManifestFile] = CoreMLEmbeddingModel.textManifest,
        session: URLSession? = nil,
    ) {
        self.modelID = modelID
        self.modelDirectory = modelDirectory
            ?? StorageRoot.directory("AskModels").appending(path: modelID, directoryHint: .isDirectory)
        self.baseURL = baseURL
        self.manifest = manifest
        if let session {
            self.session = session
            self.ownsSession = false
        } else {
            let config = URLSessionConfiguration.default
            config.timeoutIntervalForRequest = 60
            config.timeoutIntervalForResource = 1_800
            config.allowsConstrainedNetworkAccess = false
            self.session = URLSession(configuration: config)
            self.ownsSession = true
        }
    }

    deinit {
        if ownsSession {
            session.finishTasksAndInvalidate()
        }
    }

    // MARK: - Installation & Storage

    /// Fast startup check verifying every manifest file exists with its exact
    /// expected byte count (full SHA-256 runs at install time; hashing 313 MB
    /// on every launch would burn battery and stall reader open).
    public nonisolated func isInstalledOnDisk() -> Bool {
        let fm = FileManager.default
        for item in manifest {
            let url = modelDirectory.appending(path: item.relativePath)
            guard let attrs = try? fm.attributesOfItem(atPath: url.path(percentEncoded: false)),
                  let size = (attrs[.size] as? NSNumber)?.int64Value,
                  size == item.sizeBytes
            else {
                return false
            }
        }
        return true
    }

    /// Removes any abandoned `.downloading-*` staging directories left behind by
    /// an earlier crash or system kill so disk space is never leaked.
    public nonisolated func sweepIncompleteDownloads() {
        let fm = FileManager.default
        let parent = modelDirectory.deletingLastPathComponent()
        let prefix = "\(modelDirectory.lastPathComponent)\(Self.stagingPrefix)"
        guard let entries = try? fm.contentsOfDirectory(
            at: parent, includingPropertiesForKeys: nil,
        ) else { return }
        for entry in entries where entry.lastPathComponent.hasPrefix(prefix) {
            try? fm.removeItem(at: entry)
        }
    }

    /// Downloads all text-only model files using chunked `URLSession.download`
    /// into a staging folder, verifies each file's SHA-256 in 1 MB chunks,
    /// excludes the folder from iCloud backup, and atomically installs it.
    public func downloadModel(
        progress: (@Sendable (Double) -> Void)? = nil,
    ) async throws {
        sweepIncompleteDownloads()

        let parent = modelDirectory.deletingLastPathComponent()
        let fm = FileManager.default
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)
        Self.excludeFromBackup(parent)

        let totalBytes = max(Int64(1), manifest.reduce(0) { $0 + $1.sizeBytes })
        if let freeSpace = Self.availableDiskSpace(at: parent),
           freeSpace < totalBytes + Limits.freeSpaceHeadroomBytes {
            throw ModelError.insufficientStorage
        }

        let staging = parent.appending(
            path: "\(modelDirectory.lastPathComponent)\(Self.stagingPrefix)\(UUID().uuidString)",
            directoryHint: .isDirectory,
        )
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        Self.excludeFromBackup(staging)
        defer { try? fm.removeItem(at: staging) }

        // Run network I/O and 1 MB SHA-256 hashing off-actor so `unloadModels()`
        // and cancellation remain immediately responsive during a 313 MB transfer.
        try await Self.downloadManifest(
            manifest,
            baseURL: baseURL,
            into: staging,
            totalBytes: totalBytes,
            session: session,
            progress: progress,
        )

        try Task.checkCancellation()
        unloadAll()

        if fm.fileExists(atPath: modelDirectory.path(percentEncoded: false)) {
            _ = try fm.replaceItemAt(modelDirectory, withItemAt: staging)
        } else {
            try fm.moveItem(at: staging, to: modelDirectory)
        }
        Self.excludeFromBackup(modelDirectory)
        progress?(1.0)
        IssaLog.info("ask embedding model installed", ["model": modelID])
    }

    /// Unloads all in-memory handles and deletes the downloaded model files.
    public func removeModel() {
        unloadAll()
        sweepIncompleteDownloads()
        try? FileManager.default.removeItem(at: modelDirectory)
        IssaLog.info("ask embedding model removed", ["model": modelID])
    }

    /// Drops CoreML bucket handles and the memory-mapped token table to free RAM
    /// on backgrounding, idle timeout, or memory warnings while retaining the
    /// compact scalar tokeniser (~16 MB) so returning does not re-parse 32 MB JSON.
    public func unloadModels() {
        modelUnloadGeneration &+= 1
        tableLoadTask?.cancel()
        tableLoadTask = nil
        for task in bucketLoadTasks.values { task.cancel() }
        bucketLoadTasks.removeAll()
        loadedBuckets.removeAll()
        bucketUseOrder.removeAll()
        tokenTableData = nil
        rowScales = nil
    }

    /// Releases every cached resource, including the tokeniser.
    public func unloadAll() {
        unloadModels()
        tokenizerUnloadGeneration &+= 1
        tokenizerLoadTask?.cancel()
        tokenizerLoadTask = nil
        tokenizer = nil
    }

    /// True when the tokeniser, token table, and resident `embed_512` handle are
    /// all currently loaded in memory.
    public func isModelResident() -> Bool {
        let bucket = Self.bucket(forLength: Limits.maxSequenceLength)
        return tokenizer != nil
            && tokenTableData != nil
            && rowScales != nil
            && loadedBuckets[bucket] != nil
    }

    /// Asynchronously preloads the tokeniser, token table, and resident
    /// `embed_512` CoreML function off-actor so live queries do not stall on
    /// cold load/compile. Returns `true` when resident.
    @discardableResult
    public func prewarm() async -> Bool {
        guard isInstalledOnDisk() else { return false }
        let bucket = Self.bucket(forLength: Limits.maxSequenceLength)
        if isModelResident() { return true }
        guard !Task.isCancelled else { return false }
        do {
            _ = try await ensureRuntimeLoaded()
            guard !Task.isCancelled else { return false }
            _ = try await modelForBucket(bucket)
            return isModelResident()
        } catch is CancellationError {
            return false
        } catch {
            IssaLog.info("ask embedding prewarm skipped", [
                "kind": String(describing: type(of: error)),
            ])
            return false
        }
    }

    // MARK: - EmbeddingModel

    public func embed(_ texts: [String], mode: EmbeddingMode) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        guard isInstalledOnDisk() else { throw ModelError.modelNotInstalled }

        let runtime = try await ensureRuntimeLoaded()
        let prefix = mode == .query ? Self.queryPrefix : Self.documentPrefix
        var results: [[Float]] = []
        results.reserveCapacity(texts.count)

        // Reuse one pair of scratch `MLMultiArray` buffers across the batch so
        // an 8-span burst doesn't allocate/zero 4 MB of transient IOSurfaces.
        let defaultBucket = Self.bucket(forLength: Limits.maxSequenceLength)
        var scratch = try Self.makeScratchBuffers(bucket: defaultBucket)

        for text in texts {
            try Task.checkCancellation()
            let ids = runtime.tokenizer.encode(prefix + text, maxLength: Limits.maxSequenceLength)
            let bucket = Self.bucket(forLength: ids.count)
            let model = try await modelForBucket(bucket)
            if scratch.bucket != bucket {
                scratch = try Self.makeScratchBuffers(bucket: bucket)
            }
            let vector = try await Self.runSingle(
                tokenIDs: ids,
                unkTokenID: Int(runtime.tokenizer.unkTokenID),
                scratch: &scratch,
                model: model,
                tableData: runtime.tableData,
                scales: runtime.scales,
            )
            results.append(vector)
        }
        return results
    }

    // MARK: - Private Runtime Helpers

    private struct LoadedRuntime: Sendable {
        let tokenizer: GemmaTokenizer
        let tableData: Data
        let scales: [Float]
    }

    private struct ScratchBuffers {
        let bucket: Int
        let embeds: MLMultiArray
        let mask: MLMultiArray
        var previousSeqLength: Int
    }

    static func bucket(forLength count: Int) -> Int {
        for size in Limits.supportedBuckets where count <= size {
            return size
        }
        return Limits.maxSequenceLength
    }

    private func ensureRuntimeLoaded() async throws -> LoadedRuntime {
        let loadedTokenizer = try await loadTokenizer()
        let (tableData, scales) = try await loadTokenTable()
        return LoadedRuntime(tokenizer: loadedTokenizer, tableData: tableData, scales: scales)
    }

    private func loadTokenizer() async throws -> GemmaTokenizer {
        if let tokenizer { return tokenizer }
        let generation = tokenizerUnloadGeneration
        let task: Task<GemmaTokenizer, any Error>
        if let existing = tokenizerLoadTask {
            task = existing
        } else {
            let url = modelDirectory.appending(path: "text/tokenizer.json")
            let modelID = self.modelID
            let created = Task<GemmaTokenizer, any Error> {
                try await Self.loadTokenizerOffActor(from: url, modelID: modelID)
            }
            tokenizerLoadTask = created
            task = created
        }
        let loaded: GemmaTokenizer
        do {
            loaded = try await task.value
        } catch {
            if tokenizerUnloadGeneration == generation {
                tokenizerLoadTask = nil
            }
            throw error
        }
        guard tokenizerUnloadGeneration == generation else {
            throw CancellationError()
        }
        tokenizer = loaded
        tokenizerLoadTask = nil
        return loaded
    }

    private nonisolated static func loadTokenizerOffActor(
        from url: URL, modelID: String,
    ) async throws -> GemmaTokenizer {
        let clock = ContinuousClock()
        let start = clock.now
        let parsed = try GemmaTokenizer(contentsOf: url)
        IssaLog.info("ask embedding runtime loaded", [
            "model": modelID,
            "tokenizerMs": String(milliseconds(since: start, on: clock)),
        ])
        return parsed
    }

    private func loadTokenTable() async throws -> (Data, [Float]) {
        if let tokenTableData, let rowScales {
            return (tokenTableData, rowScales)
        }
        let generation = modelUnloadGeneration
        let task: Task<(Data, [Float]), any Error>
        if let existing = tableLoadTask {
            task = existing
        } else {
            let tableURL = modelDirectory.appending(path: "text/embeddings.int8")
            let scaleURL = modelDirectory.appending(path: "text/embeddings.int8.scale.f16")
            let modelID = self.modelID
            let created = Task<(Data, [Float]), any Error> {
                try await Self.loadTokenTableOffActor(
                    tableURL: tableURL, scaleURL: scaleURL, modelID: modelID,
                )
            }
            tableLoadTask = created
            task = created
        }
        let loaded: (Data, [Float])
        do {
            loaded = try await task.value
        } catch {
            if modelUnloadGeneration == generation {
                tableLoadTask = nil
            }
            throw error
        }
        guard modelUnloadGeneration == generation else {
            throw CancellationError()
        }
        tokenTableData = loaded.0
        rowScales = loaded.1
        tableLoadTask = nil
        return loaded
    }

    private nonisolated static func loadTokenTableOffActor(
        tableURL: URL, scaleURL: URL, modelID: String,
    ) async throws -> (Data, [Float]) {
        let clock = ContinuousClock()
        let start = clock.now
        let mapped = try Data(contentsOf: tableURL, options: .mappedIfSafe)
        let expectedTableBytes = Limits.vocabSize * Limits.hiddenSize
        guard mapped.count == expectedTableBytes else {
            throw ModelError.corruptAsset
        }
        let raw = try Data(contentsOf: scaleURL)
        let expectedScaleBytes = Limits.vocabSize * MemoryLayout<UInt16>.size
        guard raw.count == expectedScaleBytes else {
            throw ModelError.corruptAsset
        }
        let scales = raw.withUnsafeBytes { buffer in
            var out = [Float](repeating: 0, count: Limits.vocabSize)
            for index in 0..<Limits.vocabSize {
                let bits = buffer.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                let half = EmbeddingMath.HalfFloat.toFloat(UInt16(littleEndian: bits))
                out[index] = half * Limits.embedScale
            }
            return out
        }
        IssaLog.info("ask embedding runtime loaded", [
            "model": modelID,
            "tableMs": String(milliseconds(since: start, on: clock)),
        ])
        return (mapped, scales)
    }

    private func modelForBucket(_ bucket: Int) async throws -> SendableMLModel {
        if let existing = loadedBuckets[bucket] {
            if let position = bucketUseOrder.firstIndex(of: bucket) {
                bucketUseOrder.remove(at: position)
            }
            bucketUseOrder.append(bucket)
            return existing
        }

        let generation = modelUnloadGeneration
        let task: Task<SendableMLModel, any Error>
        if let existingTask = bucketLoadTasks[bucket] {
            task = existingTask
        } else {
            let compiledURL = modelDirectory.appending(
                path: "text/EmbeddingGemma2Text.mlmodelc",
                directoryHint: .isDirectory,
            )
            let created = Task<SendableMLModel, any Error> {
                try await Self.loadBucketOffActor(compiledURL: compiledURL, bucket: bucket)
            }
            bucketLoadTasks[bucket] = created
            task = created
        }

        let loaded: SendableMLModel
        do {
            loaded = try await task.value
        } catch {
            if modelUnloadGeneration == generation {
                bucketLoadTasks.removeValue(forKey: bucket)
            }
            throw error
        }

        guard modelUnloadGeneration == generation else {
            throw CancellationError()
        }

        while loadedBuckets.count >= Limits.maxLoadedBuckets, let oldest = bucketUseOrder.first {
            bucketUseOrder.removeFirst()
            loadedBuckets.removeValue(forKey: oldest)
        }
        loadedBuckets[bucket] = loaded
        bucketUseOrder.append(bucket)
        bucketLoadTasks.removeValue(forKey: bucket)
        return loaded
    }

    private nonisolated static func loadBucketOffActor(
        compiledURL: URL, bucket: Int,
    ) async throws -> SendableMLModel {
        let clock = ContinuousClock()
        let start = clock.now
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        config.functionName = "embed_\(bucket)"
        let loaded = try await MLModel.load(contentsOf: compiledURL, configuration: config)
        IssaLog.info("ask embedding bucket loaded", [
            "bucket": String(bucket),
            "ms": String(milliseconds(since: start, on: clock)),
        ])
        return SendableMLModel(model: loaded)
    }

    private static func milliseconds(since start: ContinuousClock.Instant, on clock: ContinuousClock) -> Int {
        let elapsed = start.duration(to: clock.now)
        return Int(elapsed.components.seconds * 1_000 + elapsed.components.attoseconds / 1_000_000_000_000_000)
    }

    private nonisolated static func makeScratchBuffers(bucket: Int) throws -> ScratchBuffers {
        let embeds = try MLMultiArray(
            shape: [1, NSNumber(value: bucket), NSNumber(value: Limits.hiddenSize)],
            dataType: .float16,
        )
        let mask = try MLMultiArray(
            shape: [1, NSNumber(value: bucket)],
            dataType: .float16,
        )
        embeds.withUnsafeMutableBytes { rawBuffer, _ in
            _ = rawBuffer.initializeMemory(as: UInt8.self, repeating: 0)
        }
        mask.withUnsafeMutableBytes { rawBuffer, _ in
            _ = rawBuffer.initializeMemory(as: UInt8.self, repeating: 0)
        }
        return ScratchBuffers(bucket: bucket, embeds: embeds, mask: mask, previousSeqLength: 0)
    }

    private nonisolated static func runSingle(
        tokenIDs: [Int32],
        unkTokenID: Int,
        scratch: inout ScratchBuffers,
        model: SendableMLModel,
        tableData: Data,
        scales: [Float],
    ) async throws -> [Float] {
        let bucket = scratch.bucket
        let seqLength = min(tokenIDs.count, bucket)
        let prevLength = scratch.previousSeqLength
        let oneBitPattern = EmbeddingMath.HalfFloat.oneBitPattern
        let safeUnk = (unkTokenID >= 0 && unkTokenID < Limits.vocabSize) ? unkTokenID : 0

        // Lock underlying buffers via `withUnsafeMutableBytes` and index with
        // CoreML's reported strides so ANE IOSurface padding is always respected.
        var embedsValid = false
        scratch.embeds.withUnsafeMutableBytes { rawBuffer, strides in
            guard let base = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt16.self),
                  strides.count >= 3
            else { return }
            let rowStride = strides[1]
            let colStride = strides[2]

            tableData.withUnsafeBytes { tableRaw in
                guard let tableBase = tableRaw.baseAddress?.assumingMemoryBound(to: Int8.self) else { return }
                for pos in 0..<seqLength {
                    let rawID = Int(tokenIDs[pos])
                    let id = (rawID >= 0 && rawID < Limits.vocabSize) ? rawID : safeUnk
                    let scale = scales[id]
                    let srcRow = id * Limits.hiddenSize
                    let dstRow = pos * rowStride
                    for col in 0..<Limits.hiddenSize {
                        let scaled = Float(tableBase[srcRow + col]) * scale
                        base[dstRow + col * colStride] = EmbeddingMath.HalfFloat.fromFloat(scaled)
                    }
                }
                if seqLength < prevLength {
                    for pos in seqLength..<prevLength {
                        let dstRow = pos * rowStride
                        for col in 0..<Limits.hiddenSize {
                            base[dstRow + col * colStride] = 0
                        }
                    }
                }
                embedsValid = true
            }
        }
        guard embedsValid else { throw ModelError.inferenceFailed }

        var maskValid = false
        scratch.mask.withUnsafeMutableBytes { rawBuffer, strides in
            guard let base = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt16.self),
                  strides.count >= 2
            else { return }
            let posStride = strides[1]
            for pos in 0..<seqLength {
                base[pos * posStride] = oneBitPattern
            }
            if seqLength < prevLength {
                for pos in seqLength..<prevLength {
                    base[pos * posStride] = 0
                }
            }
            maskValid = true
        }
        guard maskValid else { throw ModelError.inferenceFailed }
        scratch.previousSeqLength = seqLength

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "inputs_embeds": MLFeatureValue(multiArray: scratch.embeds),
            "attention_mask": MLFeatureValue(multiArray: scratch.mask),
        ])
        let output = try await model.model.prediction(from: input)
        guard let embArray = output.featureValue(for: "embedding")?.multiArrayValue else {
            throw ModelError.inferenceFailed
        }

        let dim = min(Limits.embeddingDim, embArray.count)
        guard dim == Limits.embeddingDim else { throw ModelError.inferenceFailed }
        let elemStride = embArray.strides.last?.intValue ?? 1
        var rawVector = [Float](repeating: 0, count: dim)

        var outputValid = false
        embArray.withUnsafeBytes { rawBuffer in
            switch embArray.dataType {
            case .float16:
                guard let ptr = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt16.self) else { return }
                for index in 0..<dim {
                    rawVector[index] = EmbeddingMath.HalfFloat.toFloat(ptr[index * elemStride])
                }
                outputValid = true
            case .float32:
                guard let ptr = rawBuffer.baseAddress?.assumingMemoryBound(to: Float.self) else { return }
                for index in 0..<dim {
                    rawVector[index] = ptr[index * elemStride]
                }
                outputValid = true
            default:
                for index in 0..<dim {
                    rawVector[index] = embArray[index].floatValue
                }
                outputValid = true
            }
        }
        guard outputValid else { throw ModelError.inferenceFailed }

        let normalized = EmbeddingMath.l2Normalize(rawVector)
        guard Self.isValidVector(normalized) else { throw ModelError.inferenceFailed }
        return normalized
    }

    static func isValidVector(_ vector: [Float]) -> Bool {
        guard !vector.isEmpty else { return false }
        var sumSq: Float = 0
        for value in vector {
            guard value.isFinite else { return false }
            sumSq += value * value
        }
        return sumSq > 1e-6
    }

    // MARK: - Off-Actor Download & Verification

    private static func downloadManifest(
        _ manifest: [ManifestFile],
        baseURL: URL,
        into staging: URL,
        totalBytes: Int64,
        session: URLSession,
        progress: (@Sendable (Double) -> Void)?,
    ) async throws {
        let fm = FileManager.default
        var completedBytes: Int64 = 0
        progress?(0)

        for item in manifest {
            try Task.checkCancellation()
            let remoteURL = baseURL.appending(path: item.relativePath)
            let targetURL = staging.appending(path: item.relativePath)
            try fm.createDirectory(at: targetURL.deletingLastPathComponent(), withIntermediateDirectories: true)

            let baseCompleted = completedBytes
            let delegate = DownloadProgressDelegate { written in
                let current = min(totalBytes, baseCompleted + written)
                progress?(min(0.99, Double(current) / Double(totalBytes)))
            }

            let request = URLRequest(url: remoteURL)
            let (tempFileURL, response) = try await session.download(for: request, delegate: delegate)
            defer { try? fm.removeItem(at: tempFileURL) }

            if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                throw ModelError.downloadFailed(statusCode: http.statusCode)
            }

            try Task.checkCancellation()
            let verified = try verifyFile(at: tempFileURL, expectedSize: item.sizeBytes, expectedSHA256: item.sha256)
            guard verified else {
                throw ModelError.checksumMismatch
            }

            if fm.fileExists(atPath: targetURL.path(percentEncoded: false)) {
                try fm.removeItem(at: targetURL)
            }
            try fm.moveItem(at: tempFileURL, to: targetURL)

            completedBytes += item.sizeBytes
            progress?(min(0.99, Double(completedBytes) / Double(totalBytes)))
        }
    }

    private static func verifyFile(at url: URL, expectedSize: Int64, expectedSHA256: String) throws -> Bool {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path(percentEncoded: false))
        guard let actualSize = (attrs[.size] as? NSNumber)?.int64Value,
              actualSize == expectedSize
        else { return false }

        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var hasher = SHA256()
        while true {
            try Task.checkCancellation()
            let chunk: Data? = try autoreleasepool {
                try handle.read(upToCount: Limits.hashChunkBytes)
            }
            guard let chunk, !chunk.isEmpty else { break }
            hasher.update(data: chunk)
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        return digest == expectedSHA256.lowercased()
    }

    private static func availableDiskSpace(at url: URL) -> Int64? {
        let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    private static func excludeFromBackup(_ url: URL) {
        var mutable = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? mutable.setResourceValues(values)
    }
}

private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, Sendable {
    private let onBytesWritten: @Sendable (Int64) -> Void

    init(onBytesWritten: @escaping @Sendable (Int64) -> Void) {
        self.onBytesWritten = onBytesWritten
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64,
    ) {
        onBytesWritten(totalBytesWritten)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL,
    ) {
        // Async `session.download(for:delegate:)` delivers `location` directly.
    }
}
#endif
