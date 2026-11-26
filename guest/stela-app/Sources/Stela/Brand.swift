// Reuse the repository's canonical logo in both packaged and development builds.
#if os(macOS)
  import AppKit

  enum Brand {
    static let icon: NSImage = {
      let bundled = Bundle.main.url(forResource: "stela-icon", withExtension: "png")
      let development = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().appendingPathComponent("../../../../assets/northwood-nw.png")
        .standardizedFileURL
      return bundled.flatMap(NSImage.init(contentsOf:))
        ?? NSImage(contentsOf: development)
        ?? NSImage(systemSymbolName: "cpu", accessibilityDescription: "Stela")!
    }()
  }
#endif
