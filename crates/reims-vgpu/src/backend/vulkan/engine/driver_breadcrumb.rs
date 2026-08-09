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
//!
//! # The other half: a module the validator refused
//!
//! [`keep_rejected_module`] lives here because it answers the same question with
//! the same mechanism, for the failure one step earlier. The breadcrumb catches
//! a module the *driver* could not survive; that one catches a module this
//! device assembled and `spirv-val` would not accept — which is a defect in the
//! translation rather than in the driver, and is equally unreadable from a log
//! line alone. The difference is that its file is **kept**: the event has
//! already happened when it is written, so there is nothing to disarm.

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

/// Keep a module the validator refused, beside the line that refused it.
///
/// The same argument as the breadcrumb above and a different failure. A module
/// this device assembled and `spirv-val` rejected is a **device defect**: the
/// guest asked for something translatable and the translation came out invalid.
/// The refusal line carries the validator's one-sentence complaint, and that
/// sentence names an instruction by result id — `%214 = OpCompositeInsert
/// %_struct_52 %101 %56 0 0` — which is unreadable without the module those ids
/// belong to. Reconstructing it means re-running the boot with a probe, which is
/// two rebuilds and a guest login to recover evidence the device was holding.
///
/// Unlike the breadcrumb this file is **kept**, because the event it records has
/// already happened by the time it is written. Named by content digest so
/// several distinct bad modules in one boot do not overwrite each other, and so
/// the name matches nothing else if the same module is refused again — the
/// validator's verdict is cached per digest, so it is written once per distinct
/// module per boot regardless of how many pipelines want it.
///
/// Disassemble with `spirv-dis`. A failed write costs a diagnostic and nothing
/// else; the refusal has already been emitted by the caller.
pub(crate) fn keep_rejected_module(digest: &str, spirv: &[u32]) {
    let mut bytes = Vec::with_capacity(spirv.len() * 4);
    for word in spirv {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let path = std::env::temp_dir().join(format!("reims-vgpu-rejected-{digest}.spv"));
    match std::fs::write(&path, &bytes) {
        Ok(()) => crate::observe::off(format!(
            "spirv_rejected_module_kept path={} words={}",
            path.display(),
            spirv.len()
        )),
        Err(e) => crate::observe::fail(format!(
            "spirv_rejected_module reason=write_failed path={} err={e}",
            path.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    /// The module reaches disk, little-endian and whole.
    ///
    /// Worth a test rather than trusting the write: the point of the file is to
    /// be handed to `spirv-dis`, and a word order or a truncation that made it
    /// undisassemblable would show up only on the boot where somebody needed it.
    #[test]
    fn a_rejected_module_reaches_disk_word_for_word() {
        // SPIR-V magic first, so the bytes on disk are a plausible module and a
        // reversed word order is visible rather than merely different.
        let words = [0x0723_0203u32, 0x0001_0000, 42, 0xdead_beef];
        let digest = "test0123456789ab";
        let path = std::env::temp_dir().join(format!("reims-vgpu-rejected-{digest}.spv"));
        let _ = std::fs::remove_file(&path);

        super::keep_rejected_module(digest, &words);

        let got = std::fs::read(&path).expect("the rejected module is kept");
        let _ = std::fs::remove_file(&path);
        let want: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(got, want, "little-endian words, in order, nothing added");
        assert_eq!(&got[..4], &[0x03, 0x02, 0x23, 0x07], "reads as SPIR-V magic");
    }
}
