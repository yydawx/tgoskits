//! GPIO driver using sg200x-bsp HAL.
//!
//! /dev/gpio/<name>/value     — read/write 4 bytes (32-bit port)
//! /dev/gpio/<name>/direction — read/write 4 bytes (1=output, 0=input)

use alloc::sync::Arc;
use core::any::Any;
use ax_kspin::SpinNoIrq;
use axfs_ng_vfs::{DeviceId, NodeFlags, NodeType, VfsResult};
use crate::pseudofs::{Device, DeviceOps, DirMaker, DirMapping, SimpleDir, SimpleFs};
use sg200x_bsp::gpio::GPIO;
// SAFETY: single-core SG2002, MMIO access is thread-safe
struct SyncGpio(GPIO);
unsafe impl Send for SyncGpio {}
unsafe impl Sync for SyncGpio {}
use sg200x_bsp::soc::{
    GPIO0_BASE, GPIO1_BASE, GPIO2_BASE, GPIO3_BASE, RTCSYS_GPIO_BASE,
};

const VIRT_OFFSET: usize = ax_config::plat::PHYS_VIRT_OFFSET as usize;

const BASES: &[(usize, &str)] = &[
    (GPIO0_BASE + VIRT_OFFSET, "gpio0"), (GPIO1_BASE + VIRT_OFFSET, "gpio1"),
    (GPIO2_BASE + VIRT_OFFSET, "gpio2"), (GPIO3_BASE + VIRT_OFFSET, "gpio3"),
    (RTCSYS_GPIO_BASE + VIRT_OFFSET, "gpio4"),
];

struct GpioChip(SpinNoIrq<SyncGpio>);

/// /dev/gpio/<name>/value
struct GpioVal(Arc<GpioChip>);
impl DeviceOps for GpioVal {
    fn read_at(&self, buf: &mut [u8], _off: u64) -> VfsResult<usize> {
        let v = self.0.0.lock().0.read_port().to_le_bytes();
        let n = buf.len().min(4);
        buf[..n].copy_from_slice(&v[..n]); Ok(n)
    }
    fn write_at(&self, buf: &[u8], _off: u64) -> VfsResult<usize> {
        let mut b = [0u8; 4]; let n = buf.len().min(4);
        b[..n].copy_from_slice(&buf[..n]);
        self.0.0.lock().0.write_port(u32::from_le_bytes(b)); Ok(n)
    }
    fn as_any(&self) -> &dyn Any { self }
    fn flags(&self) -> NodeFlags { NodeFlags::STREAM }
}

/// /dev/gpio/<name>/direction
struct GpioDir(Arc<GpioChip>);
impl DeviceOps for GpioDir {
    fn read_at(&self, buf: &mut [u8], _off: u64) -> VfsResult<usize> {
        let v = self.0.0.lock().0.get_direction().to_le_bytes();
        let n = buf.len().min(4);
        buf[..n].copy_from_slice(&v[..n]); Ok(n)
    }
    fn write_at(&self, buf: &[u8], _off: u64) -> VfsResult<usize> {
        let mut b = [0u8; 4]; let n = buf.len().min(4);
        b[..n].copy_from_slice(&buf[..n]);
        self.0.0.lock().0.set_direction(u32::from_le_bytes(b)); Ok(n)
    }
    fn as_any(&self) -> &dyn Any { self }
    fn flags(&self) -> NodeFlags { NodeFlags::STREAM }
}

pub fn gpio_dir_maker(fs: Arc<SimpleFs>) -> DirMaker {
    let mut dir = DirMapping::new();
    for (_i, (base, name)) in BASES.iter().enumerate() {
        let chip = Arc::new(GpioChip(SpinNoIrq::new(SyncGpio(unsafe { GPIO::new(*base) }))));
        let mut port = DirMapping::new();
        port.add("value", Device::new(fs.clone(), NodeType::CharacterDevice, DeviceId::new(0,0), Arc::new(GpioVal(chip.clone()))));
        port.add("direction", Device::new(fs.clone(), NodeType::CharacterDevice, DeviceId::new(0,0), Arc::new(GpioDir(chip))));
        dir.add(*name, SimpleDir::new_maker(fs.clone(), Arc::new(port)));
    }
    SimpleDir::new_maker(fs, Arc::new(dir))
}
