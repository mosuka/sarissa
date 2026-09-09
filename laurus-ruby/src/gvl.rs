//! GVL (Global VM Lock) release helper for blocking Rust/tokio work.
//!
//! magnus 0.8 does not bind `rb_thread_call_without_gvl` (it is listed as an
//! unbound entry in magnus's own C API coverage list), so this wraps the raw
//! `rb-sys` symbol directly.

use std::any::Any;
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

/// Run `f` with Ruby's GVL released, so other Ruby threads can run while it
/// executes.
///
/// The `Send` bound is what makes this sound: every magnus handle type
/// (`Value`, `RHash`, `RArray`, ...) is `!Send` (each carries a
/// `PhantomData<*mut RBasic>` specifically to prevent this), so a closure
/// that captures one cannot satisfy `F: Send` and this function simply will
/// not compile for such a closure -- only plain Rust/engine values may cross
/// into the released-GVL region.
///
/// No unblocking function is registered (the `ubf` argument to
/// `rb_thread_call_without_gvl` is null): there was no way to interrupt a
/// `block_on`'d call before this change either, so this is not a
/// regression, but it does mean `Thread#kill`/`Thread#raise` still cannot
/// interrupt a search/write while it runs.
pub fn without_gvl<F, T>(f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    struct Payload<F, T> {
        f: Option<F>,
        result: Option<Result<T, Box<dyn Any + Send>>>,
    }

    let mut payload = Payload {
        f: Some(f),
        result: None,
    };

    unsafe extern "C" fn trampoline<F, T>(data: *mut c_void) -> *mut c_void
    where
        F: FnOnce() -> T + Send,
    {
        // SAFETY: `data` is the address of `payload` below, which outlives
        // this call -- `rb_thread_call_without_gvl` invokes the trampoline
        // synchronously before returning.
        let payload = unsafe { &mut *data.cast::<Payload<F, T>>() };
        let f = payload.f.take().expect("trampoline called more than once");
        // Catch panics here, before crossing back over the C frame boundary
        // `rb_thread_call_without_gvl` sets up -- unwinding through it is
        // undefined behavior. Re-thrown by `without_gvl` after the GVL is
        // reacquired.
        payload.result =
            Some(panic::catch_unwind(AssertUnwindSafe(f)).map_err(|e| e as Box<dyn Any + Send>));
        std::ptr::null_mut()
    }

    let data_ptr = std::ptr::from_mut(&mut payload).cast::<c_void>();
    // SAFETY: `trampoline::<F, T>` matches the C signature
    // `void *(*)(void *)`, `data_ptr` stays valid for the duration of this
    // (synchronous) call, and no unblocking function is registered.
    unsafe {
        rb_sys::rb_thread_call_without_gvl(
            Some(trampoline::<F, T>),
            data_ptr,
            None,
            std::ptr::null_mut(),
        );
    }

    match payload.result.take().expect("trampoline did not run") {
        Ok(value) => value,
        Err(panic_payload) => panic::resume_unwind(panic_payload),
    }
}
