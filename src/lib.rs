#[macro_use]
extern crate lazy_static;
extern crate libc;
extern crate mxnet_sys;

#[macro_use]
pub mod util;
pub mod ndarray;
pub mod symbol;

pub use util::{MXError, random_seed, notify_shutdown};
pub use ndarray::{Context, NDArray, NDArrayBuilder};
pub use symbol::{Symbol, SymbolBuilder, Variable, Group};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
