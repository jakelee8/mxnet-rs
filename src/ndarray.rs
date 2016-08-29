use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::mem;
use std::ops;
use std::ptr;
use std::slice;

use libc::{c_int, c_uint, c_void};
use mxnet_sys::*;
use util::*;

macro_rules! ops {
    (
        $op_name:expr,
        $op_const:ident,
        $op_const_scalar:ident,
        $op_class:ident::$op_method:ident,
        $op_assign_class:ident::$op_assign_method:ident
    ) => {
        lazy_static! {
            static ref $op_const: Op = Op(get_function($op_name));
            static ref $op_const_scalar: Op = Op(get_function(concat!($op_name, "_scalar")));
        }
        impl ops::$op_class for NDArray {
            type Output = NDArray;

            fn $op_method(self, rhs: NDArray) -> NDArray {
                let mut handle = ptr::null_mut();
                let mut input_handle = vec![self.handle, rhs.handle];
                c_must!(MXFuncInvoke($op_const.0, input_handle.as_mut_ptr(),
                                     ptr::null_mut(), &mut handle));
                NDArray { handle: handle }
            }
        }
        impl ops::$op_class<f32> for NDArray {
            type Output = NDArray;

            fn $op_method(self, mut scalar: f32) -> NDArray {
                let mut handle = ptr::null_mut();
                let mut input_handle = vec![self.handle];
                c_must!(MXFuncInvoke($op_const_scalar.0,
                                     input_handle.as_mut_ptr(),
                                     &mut scalar,
                                     &mut handle));
                NDArray { handle: handle }
            }
        }
        impl ops::$op_assign_class for NDArray {
            fn $op_assign_method(&mut self, rhs: NDArray) {
                let mut input_handle = vec![self.handle, rhs.handle];
                c_must!(MXFuncInvoke($op_const.0, input_handle.as_mut_ptr(),
                                     ptr::null_mut(), &mut self.handle));
            }
        }
        impl ops::$op_assign_class<f32> for NDArray {
            fn $op_assign_method(&mut self, mut scalar: f32) {
                let mut input_handle = vec![self.handle];
                c_must!(MXFuncInvoke($op_const_scalar.0,
                                     input_handle.as_mut_ptr(),
                                     &mut scalar,
                                     &mut self.handle));
            }
        }
    };
}

// Force Rust to compile lazy static function handlers.
struct Op(FunctionHandle);
unsafe impl Sync for Op {}

#[derive(Debug, Copy, Clone)]
pub enum DeviceType {
    CPU = 1,
    GPU = 2,
    CPUPinned = 3,
}

#[derive(Debug, Copy, Clone)]
pub struct Context {
    pub device_type: DeviceType,
    pub device_id: isize,
}

impl Default for Context {
    fn default() -> Self {
        Context::new(DeviceType::CPU, 0)
    }
}

impl Context {
    pub fn new(device_type: DeviceType, device_id: isize) -> Self {
        Context {
            device_type: device_type,
            device_id: device_id,
        }
    }

    pub fn gpu(device_id: isize) -> Self {
        Self::new(DeviceType::GPU, device_id)
    }

    pub fn default_gpu() -> Self {
        Self::gpu(0)
    }

    pub fn cpu(device_id: isize) -> Self {
        Self::new(DeviceType::CPU, device_id)
    }

    pub fn default_cpu() -> Self {
        Self::cpu(0)
    }
}

pub struct NDArrayBuilder<'a> {
    data: Option<&'a Vec<f32>>,
    shape: Vec<u32>,
    context: Context,
    delay_alloc: bool,
}

impl<'a> NDArrayBuilder<'a> {
    pub fn new(shape: Vec<u32>) -> Self {
        NDArrayBuilder {
            data: None,
            shape: shape,
            context: Default::default(),
            delay_alloc: true,
        }
    }

    pub fn from(data: &'a Vec<f32>) -> Self {
        NDArrayBuilder {
            data: Some(data),
            shape: vec![data.len() as u32],
            context: Default::default(),
            delay_alloc: true,
        }
    }

    pub fn context(&mut self, context: Context) -> &mut Self {
        self.context = context;
        self
    }

    pub fn delay_alloc(&mut self, delay: bool) -> &mut Self {
        self.delay_alloc = delay;
        self
    }

    pub fn create(&self) -> MXResult<NDArray> {
        let mut handle = ptr::null_mut();
        c_try!(MXNDArrayCreate(self.shape.as_ptr(),
                               self.shape.len() as c_uint,
                               self.context.device_type as c_int,
                               self.context.device_id as c_int,
                               (!self.data.is_some() && self.delay_alloc) as c_int,
                               &mut handle));
        match self.data {
            Some(data) => {
                c_try!(MXNDArraySyncCopyFromCPU(handle,
                                                data.as_ptr() as *const c_void,
                                                data.len() * mem::size_of::<f32>()));
            }
            _ => {}
        }
        Ok(NDArray { handle: handle })
    }
}

pub struct NDArray {
    handle: NDArrayHandle,
}

impl NDArray {
    pub fn new() -> MXResult<Self> {
        let mut handle = ptr::null_mut();
        c_try!(MXNDArrayCreateNone(&mut handle));
        Ok(NDArray { handle: handle })
    }

    pub fn from(data: &Vec<f32>) -> MXResult<Self> {
        let mut handle = ptr::null_mut();
        c_try!(MXNDArrayCreateNone(&mut handle));
        c_try!(MXNDArraySyncCopyFromCPU(handle, data.as_ptr() as *const c_void, data.len()));
        Ok(NDArray { handle: handle })
    }

    fn load_impl(file_name: &str, with_names: bool) -> MXResult<(Vec<Self>, Option<Vec<String>>)> {
        let c_file_name = CString::new(file_name).unwrap();
        let mut out_size = 0;
        let mut out_arr = ptr::null_mut();
        let mut out_name_size = 0;
        let mut out_names = ptr::null();
        let out_names_ptr = if with_names {
            &mut out_names
        } else {
            ptr::null_mut()
        };
        c_try!(MXNDArrayLoad(c_file_name.as_ptr(),
                             &mut out_size,
                             &mut out_arr,
                             &mut out_name_size,
                             out_names_ptr));

        let out_slice = unsafe { slice::from_raw_parts(out_arr, out_size as usize) };
        let mut out_vec = Vec::with_capacity(out_size as usize);
        for handle_ptr in out_slice {
            out_vec.push(NDArray { handle: *handle_ptr });
        }

        let out_names_vec = if with_names && out_name_size > 0 {
            if out_name_size != out_size {
                return Err(MXError::new("NDArray load with names size mismatch"));
            }
            let out_names_slice =
                unsafe { slice::from_raw_parts(out_names, out_name_size as usize) };
            let mut out_names_vec = Vec::with_capacity(out_name_size as usize);
            for c_name in out_names_slice {
                let name = unsafe { CStr::from_ptr(*c_name) };
                out_names_vec.push(name.to_string_lossy().into_owned());
            }
            Some(out_names_vec)
        } else {
            None
        };

        Ok((out_vec, out_names_vec))
    }

    pub fn load(file_name: &str) -> MXResult<(Vec<Self>, Option<Vec<String>>)> {
        Self::load_impl(file_name, true)
    }

    pub fn load_list(file_name: &str) -> MXResult<Vec<Self>> {
        Ok(try!(Self::load_impl(file_name, false)).0)
    }

    pub fn load_map(file_name: &str) -> MXResult<HashMap<String, Self>> {
        match try!(Self::load_impl(file_name, false)) {
            (mut arrs, Some(mut names)) => {
                let mut map = HashMap::with_capacity(arrs.len());
                while !arrs.is_empty() {
                    map.insert(names.remove(0), arrs.remove(0));
                }
                Ok(map)
            }
            _ => Err(MXError::new("NDArray load missing names")),
        }
    }

    pub fn save_list(file_name: &str, array_list: &Vec<Self>) -> MXResult<()> {
        let c_file_name = CString::new(file_name).unwrap();
        let num_args = array_list.len();
        let mut args = Vec::with_capacity(num_args);
        for arr in array_list.iter() {
            args.push(arr.handle);
        }
        c_try!(MXNDArraySave(c_file_name.as_ptr(),
                             num_args as u32,
                             args.as_mut_ptr(),
                             ptr::null()));
        Ok(())
    }

    pub fn save_map(file_name: &str, array_map: &HashMap<String, Self>) -> MXResult<()> {
        let c_file_name = CString::new(file_name).unwrap();
        let num_args = array_map.len();
        let mut args = Vec::with_capacity(num_args);
        let mut names = Vec::with_capacity(num_args);
        for (name, arr) in array_map.iter() {
            args.push(arr.handle);
            names.push(CString::new(name.as_str()).unwrap().as_ptr());
        }
        c_try!(MXNDArraySave(c_file_name.as_ptr(),
                             num_args as u32,
                             args.as_mut_ptr(),
                             names.as_ptr()));
        Ok(())
    }

    pub fn size(&self) -> usize {
        self.raw_shape().iter().fold(1, |acc, x| acc * *x as usize)
    }

    pub fn shape(&self) -> Vec<usize> {
        let shp = self.raw_shape();
        let mut ret = Vec::with_capacity(shp.len());
        for i in shp {
            ret.push(*i as usize);
        }
        ret
    }

    pub fn reshape(&self, shape: Vec<i32>) -> MXResult<Self> {
        let mut handle = ptr::null_mut();
        c_try!(MXNDArrayReshape(self.handle,
                                shape.len() as c_int,
                                shape.as_ptr(),
                                &mut handle));
        Ok(NDArray { handle: handle })
    }

    fn raw_shape(&self) -> &[mx_uint] {
        let mut out_pdata = ptr::null();
        let mut out_dim = 0;
        c_must!(MXNDArrayGetShape(self.handle, &mut out_dim, &mut out_pdata));
        unsafe { slice::from_raw_parts(out_pdata, out_dim as usize) }
    }
}

impl Drop for NDArray {
    fn drop(&mut self) {
        c_must!(MXNDArrayFree(self.handle));
    }
}

ops!("_plus",
     OP_PLUS,
     OP_PLUS_SCALAR,
     Add::add,
     AddAssign::add_assign);

ops!("_minus",
     OP_MINUS,
     OP_MINUS_SCALAR,
     Sub::sub,
     SubAssign::sub_assign);

ops!("_div",
     OP_DIV,
     OP_DIV_SCALAR,
     Div::div,
     DivAssign::div_assign);

ops!("_mul",
     OP_MUL,
     OP_MUL_SCALAR,
     Mul::mul,
     MulAssign::mul_assign);
