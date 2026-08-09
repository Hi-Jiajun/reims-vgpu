//! What the device was inside when the process died.
//!
//! # Why a file and not a log line
//!
//! A driver that segmentation-faults inside a call we made ends the VM process
//! with no unwinding, no `Drop`, and no chance to write anything afterwards. The
//! fail log names what ran *before*, which on a shader-compile crash is a
//! translation that succeeded — the module the compiler choked on is a value in
//! this process's memory and dies with it. A core dump holds it, and digging a
//! `Vec<u32>` out of a 2 GiB core is not a workflow.
//!
//! So the bytes go to disk *before* the call and are removed *after* it. On a
//! healthy boot the file exists for the duration of one `vkCreateShaderModule`
//! plus one `vkCreateComputePipelines` and nothing is left behind. On a crash it
//! is still there, and it is exactly the module that killed the process.
//!
//! # What it costs
//!
//! One write per pipeline *miss*. Pipeline creation is cached by content digest
//! and a driven boot of a macOS guest takes single-digit misses per second at
//! its very busiest and none at all once the caches are warm, so this is not on
//! any hot path. A hit writes nothing, which is why the caller asks the cache
//! first rather than arming unconditionally.
//!
//! # What it is not
//!
//! It is not a dump facility and it is not an operator switch. There is one
//! path, it is overwritten by the next arming, and its whole contract is "if
//! this file exists, the process died in a driver call with this input".

use std::path::PathBuf;

/// The one path. Fixed rather than per-pipeline so a crash leaves exactly one
/// file to look at, and so a previous boot's leftovers cannot accumulate.
fn path() -> PathBuf {
    std::env::temp_dir().join("reims-vgpu-driver-breadcrumb.spv")
}

/// The metadata file beside it, so a reader knows what the module was for
/// without disassembling it.
fn meta_path() -> PathBuf {
    std::env::temp_dir().join("reims-vgpu-driver-breadcrumb.txt")
}

/// Live for the duration of a driver call that could take the process down.
///
/// Dropping it removes the files, so a crash is the only way they survive.
pub(crate) struct DriverBreadcrumb {
    armed: bool,
}

impl DriverBreadcrumb {
    /// Write `spirv` and a one-line description, and hold them until dropped.
    ///
    /// A write that fails is reported once and leaves the guard inert: losing
    /// the breadcrumb is a lost diagnostic, never a reason to skip the work the
    /// guest asked for.
    pub(crate) fn arm(what: &str, spirv: &[u32]) -> Self {
        let mut bytes = Vec::with_capacity(spirv.len() * 4);
        for word in spirv {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        if let Err(e) = std::fs::write(path(), &bytes) {
            crate::observe::fail(format!(
                "driver_breadcrumb reason=write_failed what={what} err={e}"
            ));
            return Self { armed: false };
        }
        let _ = std::fs::write(
            meta_path(),
            format!("{what}\nwords={}\nbytes={}\n", spirv.len(), bytes.len()),
        );
        Self { armed: true }
    }

    /// The call returned, so the input is not the one that kills the process.
    pub(crate) fn disarm(mut self) {
        self.clear();
    }

    fn clear(&mut self) {
        if !self.armed {
            return;
        }
        self.armed = false;
        let _ = std::fs::remove_file(path());
        let _ = std::fs::remove_file(meta_path());
    }
}

impl Drop for DriverBreadcrumb {
    fn drop(&mut self) {
        self.clear();
    }
}
