//! macOS shortcut capture via `NSApplication.presentationOptions`.
//!
//! # Contract, and its limit
//!
//! `NSApplicationPresentationDisableProcessSwitching` stops the window server
//! from acting on Cmd+Tab and the Dock's own chords for the presenting
//! application; combined with `DisableHideApplication` it also keeps Cmd+H from
//! being taken before the guest sees it. That is the whole of what an
//! unprivileged application may claim.
//!
//! It is **not** full capture, and this file does not pretend otherwise. The
//! window server keeps a small reserved set regardless — Cmd+Space for the
//! system search field, the screenshot chords, Ctrl+F-key accessibility
//! bindings, and anything the user has bound in Keyboard Shortcuts. Taking those
//! requires a `CGEventTap` at the HID tap point, which requires Accessibility
//! permission: a user-granted, per-application entitlement that cannot be
//! obtained programmatically and would silently produce no capture at all if
//! absent.
//!
//! So this reports [`CaptureError::PartialOnly`] on the first activation. The
//! guest gets Cmd+Tab and Cmd+H; it does not get the reserved set, and the
//! operator learns that from the fail log rather than from a guest that ignores
//! a keystroke. Nothing here infers the reserved set or works around it.

use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

use super::{Capture, CaptureError};

/// `NSApplicationPresentationDisableProcessSwitching` — suppresses Cmd+Tab and
/// the Dock chords. From `NSApplication.h`.
const DISABLE_PROCESS_SWITCHING: usize = 1 << 5;
/// `NSApplicationPresentationDisableHideApplication` — suppresses Cmd+H.
const DISABLE_HIDE_APPLICATION: usize = 1 << 8;

/// Shortcut capture over the process's `NSApplication`.
///
/// There is one `NSApplication` per process and the host window is the only
/// thing in this process that presents, so this needs no window handle: the
/// presentation options are an application-wide property.
pub struct MacCapture {
    active: bool,
    /// Whether the partial-capture refusal has already been reported. It is a
    /// standing limitation, not a per-transition event, so it is logged once.
    reported_partial: bool,
}

impl MacCapture {
    pub fn new() -> Self {
        Self {
            active: false,
            reported_partial: false,
        }
    }

    /// Set `NSApp.presentationOptions`, preserving every bit this file does not
    /// own — a fullscreen presentation sets its own options through the same
    /// property, and clobbering them would take the window out of fullscreen.
    fn apply(&self, add: bool) {
        // SAFETY: `NSApp` is the process's shared application object; both the
        // getter and the setter are main-thread-only, and on this platform the
        // window's event loop *is* the process main thread (see
        // `present::run_main_thread`).
        unsafe {
            let app: *mut Object = msg_send![class!(NSApplication), sharedApplication];
            if app.is_null() {
                return;
            }
            let current: usize = msg_send![app, presentationOptions];
            let ours = DISABLE_PROCESS_SWITCHING | DISABLE_HIDE_APPLICATION;
            let next = if add { current | ours } else { current & !ours };
            let _: () = msg_send![app, setPresentationOptions: next];
        }
    }
}

impl Default for MacCapture {
    fn default() -> Self {
        Self::new()
    }
}

impl Capture for MacCapture {
    fn set(&mut self, active: bool) -> Result<(), CaptureError> {
        if active == self.active {
            return Ok(());
        }
        self.apply(active);
        self.active = active;
        if active && !self.reported_partial {
            self.reported_partial = true;
            // Say once, on the always-on channel, exactly which keystrokes the
            // guest will still not receive on this host.
            return Err(CaptureError::PartialOnly("window_server_reserved_chords"));
        }
        Ok(())
    }

    fn describe(&self) -> &'static str {
        "macos_presentation_options"
    }
}

impl Drop for MacCapture {
    fn drop(&mut self) {
        if self.active {
            self.apply(false);
        }
    }
}
