#if !os(tvOS)
import Foundation
import IssaAsk
import Testing

@testable import IssaReader_iOS

@Suite("AskEmbeddingCoordinatorTests")
@MainActor
struct AskEmbeddingCoordinatorTests {
    @Test("Settings row hides Download when Ask cannot run, but keeps Remove visible if installed")
    func settingsRowVisibility() {
        // Ask enabled & device ready -> offered.
        #expect(AskSettingsSection.offersEmbeddingModelRow(
            availability: .available, providers: [], askEnabled: true, modelState: .notInstalled,
        ))
        // Ask turned off while not installed -> hidden.
        #expect(!AskSettingsSection.offersEmbeddingModelRow(
            availability: .available, providers: [], askEnabled: false, modelState: .notInstalled,
        ))
        // Ask turned off while already installed -> stays visible so 313 MB is never stranded.
        #expect(AskSettingsSection.offersEmbeddingModelRow(
            availability: .available, providers: [], askEnabled: false, modelState: .ready,
        ))
        // Apple Intelligence off with no provider keys -> hidden when not installed.
        #expect(!AskSettingsSection.offersEmbeddingModelRow(
            availability: .appleIntelligenceOff, providers: [], askEnabled: true, modelState: .notInstalled,
        ))
        // Provider key registered on unsupported device -> offered.
        #expect(AskSettingsSection.offersEmbeddingModelRow(
            availability: .unsupportedDevice, providers: [.openAI], askEnabled: true, modelState: .notInstalled,
        ))
    }

    @Test("Settings footer discloses the one-time Hugging Face download when offered")
    func settingsNoteDisclosure() {
        let withoutRow = AskSettingsSection.note(
            for: .available, askEnabled: true, cloudOffered: false, providers: [], searchModelOffered: false,
        )
        #expect(!withoutRow.contains("Hugging Face"))

        let withRow = AskSettingsSection.note(
            for: .available, askEnabled: true, cloudOffered: false, providers: [], searchModelOffered: true,
        )
        #expect(withRow.contains("Hugging Face"))
        #expect(withRow.contains(AskDevice.noun))
    }

    @Test("Download errors translate to plain reader prose without internal paths")
    func errorMessagesStayPlain() {
        let checksum = AskCoordinator.downloadErrorMessage(for: CoreMLEmbeddingModel.ModelError.checksumMismatch)
        #expect(!checksum.contains("weights"))
        #expect(!checksum.contains(".bin"))

        let offline = AskCoordinator.downloadErrorMessage(for: URLError(.notConnectedToInternet))
        #expect(offline.contains("internet"))
    }
}
#endif
