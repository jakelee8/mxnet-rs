use libc::c_int;

use std::error;
use std::ffi::{CStr, CString};
use std::fmt;
use std::ptr;
use std::str;

use mxnet_sys::*;

#[macro_export]
macro_rules! c_must {
    ( $expr:expr ) => {{
        if unsafe { $expr } != 0 {
            panic!(get_last_error());
        }
    }};
}

#[macro_export]
macro_rules! c_try {
    ( $expr:expr ) => { c_try!($expr, {}) };
    ( $expr:expr, $ok:expr ) => {{
        if unsafe { $expr } != 0 {
            return error_result();
        } else {
            $ok
        }
    }};
}

#[derive(Debug)]
pub struct MXError {
    errmsg: &'static str,
}

impl MXError {
    pub fn new(errmsg: &'static str) -> Self {
        MXError { errmsg: errmsg }
    }
}

impl error::Error for MXError {
    fn description(&self) -> &str {
        self.errmsg
    }
}

impl fmt::Display for MXError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "mxnet error: {}", self.errmsg)
    }
}

pub type MXResult<T> = Result<T, MXError>;

pub fn get_last_error() -> &'static str {
    let errmsg = unsafe { CStr::from_ptr(MXGetLastError()) };
    errmsg.to_str().unwrap()
}

pub fn error_result<T>() -> MXResult<T> {
    Err(MXError::new(get_last_error()))
}

pub fn get_function(name: &'static str) -> FunctionHandle {
    let mut func_handle = ptr::null();
    let c_str = CString::new(name).unwrap();
    c_must!(MXGetFunction(c_str.as_ptr(), &mut func_handle));
    func_handle
}

/// Seed the global random number generators in mxnet.
pub fn random_seed(seed: isize) -> MXResult<()> {
    c_try!(MXRandomSeed(seed as c_int), Ok(()))
}

/// Notify the engine about a shutdown.
///
/// This can help engine to print fewer console messages. It is not necessary
/// to call this function.
pub fn notify_shutdown() -> MXResult<()> {
    c_try!(MXNotifyShutdown(), Ok(()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
