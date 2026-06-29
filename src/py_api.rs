use numpy::{PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray4, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::cartridge::Cartridge;
use crate::emulator::{NES_HEIGHT, NES_WIDTH};
use crate::vec_env::{MarioVecEnv, VecEnvConfig};

#[pyclass]
pub struct FastMarioVecEnv {
    inner: MarioVecEnv,
}

#[pymethods]
impl FastMarioVecEnv {
    #[new]
    #[pyo3(signature = (rom_path, num_envs, frame_skip=4, grayscale=true, frame_stack=4, terminate_on_flag=true, crop_top=0, crop_bottom=0, resize_width=84, resize_height=84, initial_state=None))]
    pub fn new(
        rom_path: String,
        num_envs: usize,
        frame_skip: usize,
        grayscale: bool,
        frame_stack: usize,
        terminate_on_flag: bool,
        crop_top: usize,
        crop_bottom: usize,
        resize_width: usize,
        resize_height: usize,
        initial_state: Option<Vec<u8>>,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be > 0"));
        }
        if frame_skip == 0 {
            return Err(PyValueError::new_err("frame_skip must be > 0"));
        }
        if frame_stack == 0 {
            return Err(PyValueError::new_err("frame_stack must be > 0"));
        }
        if crop_top + crop_bottom >= NES_HEIGHT {
            return Err(PyValueError::new_err(format!(
                "crop_top + crop_bottom must be less than {NES_HEIGHT}, got {}",
                crop_top + crop_bottom
            )));
        }
        if resize_width == 0 || resize_height == 0 {
            return Err(PyValueError::new_err(
                "resize_width and resize_height must be > 0",
            ));
        }

        let cart = Cartridge::load_ines(rom_path)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let config = VecEnvConfig {
            num_envs,
            frame_skip,
            grayscale,
            frame_stack,
            terminate_on_flag,
            crop_top,
            crop_bottom,
            resize_width,
            resize_height,
        };
        Ok(Self {
            inner: MarioVecEnv::new(cart, config, initial_state)
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
        })
    }

    #[getter]
    pub fn num_envs(&self) -> usize {
        self.inner.config().num_envs
    }

    #[getter]
    pub fn frame_skip(&self) -> usize {
        self.inner.config().frame_skip
    }

    #[getter]
    pub fn grayscale(&self) -> bool {
        self.inner.config().grayscale
    }

    #[getter]
    pub fn frame_stack(&self) -> usize {
        self.inner.config().frame_stack
    }

    #[getter]
    pub fn crop_top(&self) -> usize {
        self.inner.config().crop_top
    }

    #[getter]
    pub fn crop_bottom(&self) -> usize {
        self.inner.config().crop_bottom
    }

    #[getter]
    pub fn resize_width(&self) -> usize {
        self.inner.config().resize_width
    }

    #[getter]
    pub fn resize_height(&self) -> usize {
        self.inner.config().resize_height
    }

    pub fn obs_shape(&self) -> (usize, usize, usize, usize) {
        (
            self.inner.config().num_envs,
            self.inner.config().channels(),
            self.inner.config().obs_height(),
            self.inner.config().obs_width(),
        )
    }

    pub fn reset_into<'py>(
        &mut self,
        py: Python<'py>,
        mut obs: PyReadwriteArray4<'py, u8>,
    ) -> PyResult<()> {
        self.validate_obs_shape(&obs)?;
        let mut obs_rw = obs.as_array_mut();
        let obs_slice = obs_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("obs must be C-contiguous"))?;
        py.allow_threads(|| self.inner.reset_into(obs_slice))
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(())
    }

    pub fn info_into<'py>(
        &self,
        py: Python<'py>,
        mut x_pos: PyReadwriteArray1<'py, u16>,
        mut lives: PyReadwriteArray1<'py, u8>,
    ) -> PyResult<()> {
        self.validate_vec_len(x_pos.len(), "x_pos")?;
        self.validate_vec_len(lives.len(), "lives")?;
        let mut x_pos_rw = x_pos.as_array_mut();
        let x_pos_slice = x_pos_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("x_pos must be C-contiguous"))?;
        let mut lives_rw = lives.as_array_mut();
        let lives_slice = lives_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("lives must be C-contiguous"))?;
        py.allow_threads(|| {
            self.inner.info_into(x_pos_slice, lives_slice);
        });
        Ok(())
    }

    pub fn step_into<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'py, u8>,
        mut obs: PyReadwriteArray4<'py, u8>,
        mut rewards: PyReadwriteArray1<'py, f32>,
        mut terminated: PyReadwriteArray1<'py, bool>,
        mut truncated: PyReadwriteArray1<'py, bool>,
        mut x_pos: PyReadwriteArray1<'py, u16>,
        mut lives: PyReadwriteArray1<'py, u8>,
    ) -> PyResult<()> {
        self.validate_obs_shape(&obs)?;
        self.validate_vec_len(actions.len(), "actions")?;
        self.validate_vec_len(rewards.len(), "rewards")?;
        self.validate_vec_len(terminated.len(), "terminated")?;
        self.validate_vec_len(truncated.len(), "truncated")?;
        self.validate_vec_len(x_pos.len(), "x_pos")?;
        self.validate_vec_len(lives.len(), "lives")?;

        let actions_ro = actions.as_array();
        let actions_slice = actions_ro
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("actions must be C-contiguous"))?;
        let mut obs_rw = obs.as_array_mut();
        let obs_slice = obs_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("obs must be C-contiguous"))?;
        let mut rewards_rw = rewards.as_array_mut();
        let rewards_slice = rewards_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("rewards must be C-contiguous"))?;
        let mut terminated_rw = terminated.as_array_mut();
        let terminated_slice = terminated_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("terminated must be C-contiguous"))?;
        let mut truncated_rw = truncated.as_array_mut();
        let truncated_slice = truncated_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("truncated must be C-contiguous"))?;
        let mut x_pos_rw = x_pos.as_array_mut();
        let x_pos_slice = x_pos_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("x_pos must be C-contiguous"))?;
        let mut lives_rw = lives.as_array_mut();
        let lives_slice = lives_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("lives must be C-contiguous"))?;

        py.allow_threads(|| {
            self.inner.step_into(
                actions_slice,
                obs_slice,
                rewards_slice,
                terminated_slice,
                truncated_slice,
                x_pos_slice,
                lives_slice,
            );
        });
        Ok(())
    }
}

impl FastMarioVecEnv {
    fn validate_obs_shape(&self, obs: &PyReadwriteArray4<'_, u8>) -> PyResult<()> {
        let shape = obs.shape();
        let expected = self.obs_shape();
        if shape != [expected.0, expected.1, expected.2, expected.3] {
            return Err(PyValueError::new_err(format!(
                "obs shape must be {:?}, got {:?}",
                expected, shape
            )));
        }
        Ok(())
    }

    fn validate_vec_len(&self, len: usize, name: &str) -> PyResult<()> {
        if len != self.inner.config().num_envs {
            return Err(PyValueError::new_err(format!(
                "{name} length must be {}, got {len}",
                self.inner.config().num_envs
            )));
        }
        Ok(())
    }
}

#[pymodule]
fn _supermariobrosnes_fastenv(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastMarioVecEnv>()?;
    m.add("NES_WIDTH", NES_WIDTH)?;
    m.add("NES_HEIGHT", NES_HEIGHT)?;
    Ok(())
}
