# mxnet Rust Bindings

> This is a work in progress. Contributions gladly accepted!

The `mxnet` crate defines a high-level Rust API for [mxnet][] using the
[`mxnet-sys`][mxnet-sys] crate.

## Dependencies

To use `mxnet-sys`, you'll need to build and install `libmxnet.so` where
Cargo can find it. For example, copy `libmxnet.so` to `/usr/local/lib`.

For details on how to build `libmxnet.so`, please see the [mxnet][] project.

## License

Distributed under the [ISC License][license].

[mxnet]: https://github.com/dmlc/mxnet
[mxnet-sys]: https://github.com/jakeleeme/mxnet-sys
[license]: LICENSE.txt
