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

    @Test("EmbeddingPolicy catches up hard when behind, self-governs after 10s, and cruises cool in look-ahead")
    func embeddingPolicyPacingMatrix() {
        // 1. Live on-device Ask yields ANE immediately for both tiers.
        let liveAsk = EmbeddingPolicy.DeviceConditions(onDeviceAskRunning: true)
        #expect(EmbeddingPolicy.pace(tier: .catchUp, conditions: liveAsk) == nil)
        #expect(EmbeddingPolicy.pace(tier: .lookAhead, conditions: liveAsk) == nil)

        // 2. Nominal on battery: catch-up runs full blast for first 10s, then steps down to 50% duty.
        let nomBattery = EmbeddingPolicy.DeviceConditions(thermalState: .nominal, isCharging: false, batteryLevel: 0.75)
        let burst = EmbeddingPolicy.pace(tier: .catchUp, conditions: nomBattery)
        #expect(burst?.batchSize == 8)
        #expect(burst?.cooldown == .zero)
        #expect(burst?.priority == .utility)

        var sustained = nomBattery
        sustained.sustainedCatchUpSeconds = 12
        let governed = EmbeddingPolicy.pace(tier: .catchUp, conditions: sustained)
        #expect(governed?.batchSize == 4)
        #expect(governed?.cooldown == .milliseconds(200))

        // 3. Look-ahead on battery runs cool micro-bursts at `.background` QoS.
        let lookAhead = EmbeddingPolicy.pace(tier: .lookAhead, conditions: nomBattery)
        #expect(lookAhead?.batchSize == 4)
        #expect(lookAhead?.cooldown == .milliseconds(2_500))
        #expect(lookAhead?.priority == .background)

        // 4. Slowdown > 2x doubles look-ahead cooldown.
        var contended = nomBattery
        contended.slowdownFactor = 2.5
        #expect(EmbeddingPolicy.pace(tier: .lookAhead, conditions: contended)?.cooldown == .seconds(5))

        // 5. Charging boost accelerates look-ahead; thermal hysteresis holds look-ahead.
        let charging = EmbeddingPolicy.DeviceConditions(thermalState: .nominal, isCharging: true, batteryLevel: 0.80)
        #expect(EmbeddingPolicy.pace(tier: .lookAhead, conditions: charging)?.batchSize == 8)
        #expect(EmbeddingPolicy.pace(tier: .lookAhead, conditions: charging)?.cooldown == .milliseconds(400))

        var hysteresis = charging
        hysteresis.inThermalHysteresis = true
        #expect(EmbeddingPolicy.pace(tier: .lookAhead, conditions: hysteresis) == nil)

        // 6. Fair thermal pauses look-ahead, throttles catch-up; serious/LPM allow catch-up only when Ask sheet open.
        let fair = EmbeddingPolicy.DeviceConditions(thermalState: .fair, isCharging: false, batteryLevel: 0.80)
        #expect(EmbeddingPolicy.pace(tier: .lookAhead, conditions: fair) == nil)
        #expect(EmbeddingPolicy.pace(tier: .catchUp, conditions: fair)?.cooldown == .milliseconds(1_500))

        let lpmClosed = EmbeddingPolicy.DeviceConditions(thermalState: .nominal, isLowPowerMode: true, askSheetOpen: false)
        let lpmOpen = EmbeddingPolicy.DeviceConditions(thermalState: .nominal, isLowPowerMode: true, askSheetOpen: true)
        #expect(EmbeddingPolicy.pace(tier: .catchUp, conditions: lpmClosed) == nil)
        #expect(EmbeddingPolicy.pace(tier: .catchUp, conditions: lpmOpen)?.priority == .userInitiated)
        #expect(EmbeddingPolicy.pace(tier: .lookAhead, conditions: lpmOpen) == nil)

        // 7. Low battery (<20% unplugged) holds look-ahead, throttles catch-up.
        let lowBatt = EmbeddingPolicy.DeviceConditions(thermalState: .nominal, isCharging: false, batteryLevel: 0.15)
        #expect(EmbeddingPolicy.pace(tier: .lookAhead, conditions: lowBatt) == nil)
        #expect(EmbeddingPolicy.pace(tier: .catchUp, conditions: lowBatt)?.cooldown == .seconds(1))

        // 8. Critical always pauses both tiers, even with Ask sheet open.
        let critical = EmbeddingPolicy.DeviceConditions(thermalState: .critical, askSheetOpen: true)
        #expect(EmbeddingPolicy.pace(tier: .catchUp, conditions: critical) == nil)
        #expect(EmbeddingPolicy.pace(tier: .lookAhead, conditions: critical) == nil)
    }
}
#endif
