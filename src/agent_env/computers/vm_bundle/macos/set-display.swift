// set-display: the main display in its largest 1x mode, kept across restarts, with the sRGB colour profile.
//
// A macOS guest boots in the mode its disk was last in, not in the size Lume gives the virtual display, so a clone of
// a VM bundle composed at one size would come up at that size; this, at login, follows the virtual display. The
// virtual display's own profile makes a screenshot a colour-managed conversion of what apps drew, one off in a channel;
// with sRGB's a screenshot holds the pixels as drawn.
import ColorSync
import CoreGraphics
import Foundation

let display = CGMainDisplayID()
let uuid = CGDisplayCreateUUIDFromDisplayID(display).takeRetainedValue()
let sRGB = URL(fileURLWithPath: "/System/Library/ColorSync/Profiles/sRGB Profile.icc")
if !ColorSyncDeviceSetCustomProfiles(kColorSyncDisplayDeviceClass.takeUnretainedValue(), uuid,
                                     [kColorSyncDeviceDefaultProfileID.takeUnretainedValue() as String: sRGB] as CFDictionary) {
    FileHandle.standardError.write(Data("set-display: couldn't set the sRGB profile\n".utf8))
}
let modes = (CGDisplayCopyAllDisplayModes(display, [kCGDisplayShowDuplicateLowResolutionModes: true] as CFDictionary)
             as? [CGDisplayMode]) ?? []
let largest = modes.filter { $0.width == $0.pixelWidth && $0.height == $0.pixelHeight && $0.isUsableForDesktopGUI() }
    .max { $0.width * $0.height < $1.width * $1.height }
guard let mode = largest else {
    FileHandle.standardError.write(Data("set-display: the display offers no 1x mode\n".utf8))
    exit(1)
}
if let now = CGDisplayCopyDisplayMode(display), now.width == mode.width, now.height == mode.height {
    exit(0)
}
var config: CGDisplayConfigRef?
guard CGBeginDisplayConfiguration(&config) == .success,
      CGConfigureDisplayWithDisplayMode(config, display, mode, nil) == .success,
      CGCompleteDisplayConfiguration(config, .permanently) == .success else {
    FileHandle.standardError.write(Data("set-display: couldn't set \(mode.width)x\(mode.height)\n".utf8))
    exit(1)
}
print("set-display: \(mode.width)x\(mode.height)")
