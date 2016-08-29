use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::ops;
use std::ptr;
use std::slice;

use libc::{c_char, c_int, c_uint, c_void};
use mxnet_sys::*;
use util::*;

lazy_static! {
    static ref SYMBOL_CREATORS: HashMap<String, Creator> = {
        let mut num_symbol_creators = 0;
        let mut symbol_creators = ptr::null_mut();
        c_must!(MXSymbolListAtomicSymbolCreators(&mut num_symbol_creators, &mut symbol_creators));
        let symbol_creators_slice =
            unsafe { slice::from_raw_parts(symbol_creators, num_symbol_creators as usize) };
        let mut map = HashMap::with_capacity(num_symbol_creators as usize);
        for symbol_creator in symbol_creators_slice {
            let mut name = ptr::null();
            let mut description = ptr::null();
            let mut num_args = 0;
            let mut arg_names = ptr::null();
            let mut arg_type_infos = ptr::null();
            let mut arg_descriptions = ptr::null();
            let mut key_var_num_args = ptr::null();
            c_must!(MXSymbolGetAtomicSymbolInfo(*symbol_creator,
                                                &mut name,
                                                &mut description,
                                                &mut num_args,
                                                &mut arg_names,
                                                &mut arg_type_infos,
                                                &mut arg_descriptions,
                                                &mut key_var_num_args,
                                                ptr::null_mut()));
            let name = unsafe { CStr::from_ptr(name) };
            map.insert(name.to_string_lossy().into_owned(), Creator(*symbol_creator));
        }
        map
    };
}

// Force Rust to compile lazy static function handlers.
struct Creator(AtomicSymbolCreator);
unsafe impl Sync for Creator {}

#[derive(Debug)]
pub struct SymbolBuilder<'a> {
    operator_name: &'a str,
    input_keys: Vec<*const c_char>,
    input_values: Vec<SymbolHandle>,
    param_keys: Vec<*const c_char>,
    param_values: Vec<*const c_char>,
}

impl<'a> SymbolBuilder<'a> {
    pub fn new(operator_name: &'a str) -> Self {
        SymbolBuilder {
            operator_name: operator_name,
            input_keys: Default::default(),
            input_values: Default::default(),
            param_keys: Default::default(),
            param_values: Default::default(),
        }
    }

    pub fn add_input(&mut self, key: &str, value: &Symbol) -> &mut Self {
        self.input_keys.push(CString::new(key).unwrap().as_ptr());
        self.input_values.push(value.handle);
        self
    }

    pub fn set_input(&mut self, values: &Vec<&Symbol>) -> &mut Self {
        self.input_keys.clear();
        self.input_values = values.iter().map(|s| s.handle).collect();
        self
    }

    pub fn add_param(&mut self, key: &str, value: &str) -> &mut Self {
        self.param_keys.push(CString::new(key).unwrap().as_ptr());
        self.param_values.push(CString::new(value).unwrap().as_ptr());
        self
    }

    fn create_symbol(&self) -> MXResult<Symbol> {
        let symbol_creator = SYMBOL_CREATORS.get(self.operator_name).unwrap().0;
        let num_param = self.param_keys.len() as u32;
        let param_keys = self.param_keys.as_ptr();
        let param_values = self.param_keys.as_ptr();
        let mut handle = ptr::null_mut();
        c_try!(MXSymbolCreateAtomicSymbol(symbol_creator,
                                          num_param,
                                          param_keys,
                                          param_values,
                                          &mut handle));
        Ok(Symbol { handle: handle })
    }

    fn compose_symbol(&self, symbol: &mut Symbol, name: &str) -> MXResult<()> {
        let name = CString::new(name).unwrap().as_ptr();
        let num_input = self.input_keys.len() as u32;
        let input_keys = self.input_keys.as_ptr();
        let input_values = self.input_values.as_ptr();
        c_try!(MXSymbolCompose(symbol.handle, name, num_input, input_keys, input_values));
        Ok(())
    }

    pub fn create(&self, name: &str) -> MXResult<Symbol> {
        let mut symbol = try!(self.create_symbol());
        try!(self.compose_symbol(&mut symbol, name));
        Ok(symbol)
    }
}

pub struct Symbol {
    handle: SymbolHandle,
}

impl Drop for Symbol {
    fn drop(&mut self) {
        c_must!(MXSymbolFree(self.handle));
    }
}

impl Symbol {
    fn new(handle: SymbolHandle) -> Self {
        Symbol { handle: handle }
    }

    pub fn load(file_name: &str) -> MXResult<Self> {
        let c_file_name = CString::new(file_name).unwrap();
        let mut handle = ptr::null_mut();
        c_try!(MXSymbolCreateFromFile(c_file_name.as_ptr(), &mut handle));
        Ok(Self::new(handle))
    }

    pub fn load_json(json_str: &str) -> MXResult<Self> {
        let c_json_str = CString::new(json_str).unwrap();
        let mut handle = ptr::null_mut();
        c_try!(MXSymbolCreateFromJSON(c_json_str.as_ptr(), &mut handle));
        Ok(Self::new(handle))
    }

    pub fn save(&self, file_name: &str) -> MXResult<()> {
        let c_file_name = CString::new(file_name).unwrap();
        c_try!(MXSymbolSaveToFile(self.handle, c_file_name.as_ptr()));
        Ok(())
    }

    pub fn to_json(&self) -> MXResult<String> {
        let mut c_json_str = ptr::null();
        c_try!(MXSymbolSaveToJSON(self.handle, &mut c_json_str));
        let json_str = unsafe { CStr::from_ptr(c_json_str) };
        Ok(json_str.to_string_lossy().into_owned())
    }

    // Symbol operator+(const Symbol &rhs) const;
    // Symbol operator-(const Symbol &rhs) const;
    // Symbol operator*(const Symbol &rhs) const;
    // Symbol operator/(const Symbol &rhs) const;

    // Symbol operator+(mx_float scalar) const;
    // Symbol operator-(mx_float scalar) const;
    // Symbol operator*(mx_float scalar) const;
    // Symbol operator/(mx_float scalar) const;
    // Symbol Copy() const;

    // pub fn internals(&self) -> MXResult<Self> {}

    // pub fn list_arguments(&self) -> Vec<String> {}
    // pub fn list_outputs(&self) -> Vec<String> {}
    // pub fn list_auxilary_states(&self) -> Vec<String> {}

    pub fn output(&self, index: usize) -> MXResult<Symbol> {
        let mut handle = ptr::null_mut();
        c_try!(MXSymbolGetOutput(self.handle, index as c_uint, &mut handle));
        Ok(Self::new(handle))
    }

    // pub fn infer_shape(&self,
    //     const std::map<std::string, std::vector<mx_uint> > &arg_shapes,
    //     std::vector<std::vector<mx_uint> > *in_shape,
    //     std::vector<std::vector<mx_uint> > *aux_shape,
    //     std::vector<std::vector<mx_uint> > *out_shape) const;

    // pub fn infer_executor_arrays(
    //     const Context &context, std::vector<NDArray> *arg_arrays,
    //     std::vector<NDArray> *grad_arrays, std::vector<OpReqType> *grad_reqs,
    //     std::vector<NDArray> *aux_arrays,
    //     const std::map<std::string, NDArray> &args_map,
    //     const std::map<std::string, NDArray> &arg_grad_store =
    //         std::map<std::string, NDArray>(),
    //     const std::map<std::string, OpReqType> &grad_req_type =
    //         std::map<std::string, OpReqType>(),
    //     const std::map<std::string, NDArray> &aux_map =
    //         std::map<std::string, NDArray>()) const;

    // pub fn infer_args_map(const Context &context,
    //                   std::map<std::string, NDArray> *args_map,
    //                   const std::map<std::string, NDArray> &known_args) const;

    // pub fn simple_bind(const Context &context,
    //                      const std::map<std::string, NDArray> &args_map,
    //                      const std::map<std::string, NDArray> &arg_grad_store =
    //                          std::map<std::string, NDArray>(),
    //                      const std::map<std::string, OpReqType> &grad_req_type =
    //                          std::map<std::string, OpReqType>(),
    //                      const std::map<std::string, NDArray> &aux_map =
    //                          std::map<std::string, NDArray>()) -> Executor;

    // pub fn bind(const Context &context, const std::vector<NDArray> &arg_arrays,
    //                const std::vector<NDArray> &grad_arrays,
    //                const std::vector<OpReqType> &grad_reqs,
    //                const std::vector<NDArray> &aux_arrays,
    //                const std::map<std::string, Context> &group_to_ctx =
    //                    std::map<std::string, Context>(),
    //                Executor *shared_exec = nullptr) -> Executor;
}

// impl ops::Index<usize> for Symbol {
//     type Output = MXResult<Symbol>;
//     fn index(&self, index: usize) -> &Self::Output {
//         let handle = ptr::null_mut();
//         c_try!(MXSymbolGetOutput(self.handle, index as c_uint, &mut handle));
//         Ok(Symbol { handle: handle })
//     }
// }

pub enum Variable {}

impl Variable {
    pub fn new(name: &str) -> MXResult<Symbol> {
        let c_name = CString::new(name).unwrap();
        let mut handle = ptr::null_mut();
        c_try!(MXSymbolCreateVariable(c_name.as_ptr(), &mut handle));
        Ok(Symbol { handle: handle })
    }
}

pub enum Group {}

impl Group {
    pub fn new(symbols: Vec<Symbol>) -> MXResult<Symbol> {
        let mut handle = ptr::null_mut();
        let mut handle_list: Vec<SymbolHandle> = symbols.into_iter().map(|t| t.handle).collect();
        c_try!(MXSymbolCreateGroup(handle_list.len() as u32,
                                   handle_list.as_mut_ptr(),
                                   &mut handle));
        Ok(Symbol { handle: handle })
    }
}
