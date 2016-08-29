#[macro_use]
extern crate lazy_static;
extern crate libc;
extern crate mxnet_sys;

#[macro_use]
pub mod util;
pub mod ndarray;

pub use util::{MXError, random_seed, notify_shutdown};
pub use ndarray::{Context, NDArray, NDArrayBuilder};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
