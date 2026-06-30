use crate::cartridge::Cartridge;
use thiserror::Error;

pub const NES_WIDTH: usize = 256;
pub const NES_HEIGHT: usize = 240;
pub const VISIBLE_FRAME_LEFT: usize = 8;
pub const VISIBLE_FRAME_TOP: usize = 8;
pub const VISIBLE_FRAME_WIDTH: usize = 240;
pub const VISIBLE_FRAME_HEIGHT: usize = 224;
pub const RGB_CHANNELS: usize = 3;
#[allow(dead_code)]
pub const FRAME_PIXELS_RGB: usize = NES_WIDTH * NES_HEIGHT * RGB_CHANNELS;
#[allow(dead_code)]
pub const FRAME_PIXELS_GRAY: usize = NES_WIDTH * NES_HEIGHT;

const CPU_CYCLES_PER_FRAME_GUARD: usize = 40_000;
const PPU_DOTS_PER_SCANLINE: usize = 341;
const PPU_SCANLINES_PER_FRAME: usize = 262;
const PPU_DOTS_PER_FRAME: usize = PPU_DOTS_PER_SCANLINE * PPU_SCANLINES_PER_FRAME;
const PPU_VBLANK_DOT: usize = PPU_DOTS_PER_SCANLINE;
const PPU_PRERENDER_DOT: usize = 21 * PPU_DOTS_PER_SCANLINE;
const PPU_VISIBLE_START_SCANLINE: usize = 22;
const PPU_SPRITE0_DOT: usize = (PPU_VISIBLE_START_SCANLINE + 30) * PPU_DOTS_PER_SCANLINE + 1;
const DEFAULT_GRAY_CROP_TOP: usize = 32;
const DEFAULT_GRAY_CROP_HEIGHT: usize = VISIBLE_FRAME_HEIGHT - DEFAULT_GRAY_CROP_TOP;
const DEFAULT_GRAY_RESIZE_WIDTH: usize = 84;
const DEFAULT_GRAY_RESIZE_HEIGHT: usize = 84;
const DEFAULT_GRAY_RESIZE_PIXELS: usize = DEFAULT_GRAY_RESIZE_WIDTH * DEFAULT_GRAY_RESIZE_HEIGHT;

const FLAG_C: u8 = 0x01;
const FLAG_Z: u8 = 0x02;
const FLAG_I: u8 = 0x04;
const FLAG_D: u8 = 0x08;
const FLAG_B: u8 = 0x10;
const FLAG_U: u8 = 0x20;
const FLAG_V: u8 = 0x40;
const FLAG_N: u8 = 0x80;

const BUTTON_A: u8 = 1 << 0;
const BUTTON_B: u8 = 1 << 1;
const BUTTON_START: u8 = 1 << 3;
const BUTTON_LEFT: u8 = 1 << 6;
const BUTTON_RIGHT: u8 = 1 << 7;

#[derive(Debug, Error)]
pub enum StateLoadError {
    #[error("state field {name} with size {size} was not found")]
    MissingField { name: &'static str, size: usize },
}

#[derive(Clone, Copy, Debug)]
pub enum MarioAction {
    Noop = 0,
    Right = 1,
    RightB = 2,
    RightA = 3,
    RightAB = 4,
    A = 5,
    Left = 6,
    Start = 7,
}

impl MarioAction {
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Right,
            2 => Self::RightB,
            3 => Self::RightA,
            4 => Self::RightAB,
            5 => Self::A,
            6 => Self::Left,
            7 => Self::Start,
            _ => Self::Noop,
        }
    }

    #[inline]
    fn buttons(self) -> u8 {
        match self {
            Self::Noop => 0,
            Self::Right => BUTTON_RIGHT,
            Self::RightB => BUTTON_RIGHT | BUTTON_B,
            Self::RightA => BUTTON_RIGHT | BUTTON_A,
            Self::RightAB => BUTTON_RIGHT | BUTTON_A | BUTTON_B,
            Self::A => BUTTON_A,
            Self::Left => BUTTON_LEFT,
            Self::Start => BUTTON_START,
        }
    }
}

#[derive(Clone, Copy)]
struct Cpu {
    a: u8,
    x: u8,
    y: u8,
    sp: u8,
    pc: u16,
    p: u8,
}

impl Cpu {
    fn new() -> Self {
        Self {
            a: 0,
            x: 0,
            y: 0,
            sp: 0xfd,
            pc: 0,
            p: FLAG_U | FLAG_I,
        }
    }
}

#[derive(Clone)]
struct Ppu {
    chr_rom: Vec<u8>,
    chr_addr_mask: usize,
    vertical_mirroring: bool,
    ctrl: u8,
    mask: u8,
    status: u8,
    oam_addr: u8,
    oam: [u8; 256],
    vram: [u8; 2048],
    palette: [u8; 32],
    data_buffer: u8,
    addr: u16,
    temp_addr: u16,
    scroll_addr: u16,
    render_addr: u16,
    first_write: bool,
    fine_x: u8,
    scroll_x_px: u16,
    scroll_y_px: u16,
    scroll_x_low: u8,
    scroll_override_x_px: Option<u16>,
    frame_dot: usize,
    frame: u64,
    nmi_pending: bool,
}

impl Ppu {
    fn new(chr_rom: Vec<u8>, vertical_mirroring: bool) -> Self {
        let chr_addr_mask = chr_rom.len() - 1;
        Self {
            chr_rom,
            chr_addr_mask,
            vertical_mirroring,
            ctrl: 0,
            mask: 0,
            status: 0,
            oam_addr: 0,
            oam: [0; 256],
            vram: [0; 2048],
            palette: [0; 32],
            data_buffer: 0,
            addr: 0,
            temp_addr: 0,
            scroll_addr: 0,
            render_addr: 0,
            first_write: true,
            fine_x: 0,
            scroll_x_px: 0,
            scroll_y_px: 0,
            scroll_x_low: 0,
            scroll_override_x_px: None,
            frame_dot: 0,
            frame: 0,
            nmi_pending: false,
        }
    }

    fn reset(&mut self) {
        self.ctrl = 0;
        self.mask = 0;
        self.status = 0;
        self.oam_addr = 0;
        self.oam = [0; 256];
        self.vram = [0; 2048];
        self.palette = [0; 32];
        self.data_buffer = 0;
        self.addr = 0;
        self.temp_addr = 0;
        self.scroll_addr = 0;
        self.render_addr = 0;
        self.first_write = true;
        self.fine_x = 0;
        self.scroll_x_px = 0;
        self.scroll_y_px = 0;
        self.scroll_x_low = 0;
        self.scroll_override_x_px = None;
        self.frame_dot = 0;
        self.frame = 0;
        self.nmi_pending = false;
    }

    fn oam(&self) -> &[u8; 256] {
        &self.oam
    }

    fn debug_bg_pixel(&self, x: usize, y: usize) -> (u8, bool) {
        (self.bg_color_index(x, y), self.bg_pixel_opaque(x, y))
    }

    fn load_fceu_state(
        &mut self,
        ntar: &[u8],
        pram: &[u8],
        spra: &[u8],
        ppur: &[u8],
        radd: Option<&[u8]>,
        tadd: Option<&[u8]>,
        xoff: Option<&[u8]>,
    ) {
        self.vram.copy_from_slice(ntar);
        self.palette.copy_from_slice(pram);
        self.oam.copy_from_slice(spra);
        self.ctrl = ppur[0];
        self.mask = ppur[1];
        self.status = ppur[2];
        self.oam_addr = ppur[3];
        self.addr = radd.and_then(read_u16_le).unwrap_or(0);
        self.temp_addr = tadd.and_then(read_u16_le).unwrap_or(0);
        self.scroll_addr = self.temp_addr;
        self.render_addr = self.addr;
        self.first_write = true;
        self.fine_x = xoff.and_then(|value| value.first().copied()).unwrap_or(0);
        self.frame_dot = 0;
        self.nmi_pending = false;
        self.update_scroll_x_px();
    }

    #[inline]
    fn tick(&mut self, ppu_cycles: usize) -> bool {
        let mut completed_frame = false;
        let mut remaining = ppu_cycles;
        while remaining > 0 {
            let current = self.dot();
            let next = next_ppu_event_dot(current);
            let advance = remaining.min(next - current);
            self.set_dot(current + advance);
            remaining -= advance;

            match self.dot() {
                PPU_SPRITE0_DOT => self.status |= 0x40,
                PPU_VBLANK_DOT => {
                    self.status |= 0x80;
                    if self.ctrl & 0x80 != 0 {
                        self.nmi_pending = true;
                    }
                }
                PPU_PRERENDER_DOT => {
                    self.status &= !0xc0;
                }
                PPU_DOTS_PER_FRAME => {
                    self.frame_dot = 0;
                    self.frame = self.frame.wrapping_add(1);
                    completed_frame = true;
                }
                _ => {}
            }
        }
        completed_frame
    }

    #[inline]
    fn dot(&self) -> usize {
        self.frame_dot
    }

    #[inline]
    fn cycles_until_next_event(&self) -> usize {
        next_ppu_event_dot(self.dot()) - self.dot()
    }

    #[inline]
    fn set_dot(&mut self, dot: usize) {
        self.frame_dot = dot;
    }

    #[inline]
    fn take_nmi(&mut self) -> bool {
        let pending = self.nmi_pending;
        self.nmi_pending = false;
        pending
    }

    #[inline]
    fn cpu_read_register(&mut self, reg: u16) -> u8 {
        match reg & 7 {
            2 => {
                let value = self.status;
                self.status &= !0x80;
                self.first_write = true;
                value
            }
            4 => self.oam[self.oam_addr as usize],
            7 => self.read_data(),
            _ => 0,
        }
    }

    #[inline]
    fn cpu_write_register(&mut self, reg: u16, value: u8) {
        match reg & 7 {
            0 => {
                let old = self.ctrl;
                self.ctrl = value;
                self.temp_addr = (self.temp_addr & 0xf3ff) | (((value as u16) & 0x03) << 10);
                self.scroll_addr = (self.scroll_addr & 0xf3ff) | (((value as u16) & 0x03) << 10);
                self.update_scroll_x_px();
                if old & 0x80 == 0 && value & 0x80 != 0 && self.status & 0x80 != 0 {
                    self.nmi_pending = true;
                }
            }
            1 => self.mask = value,
            3 => self.oam_addr = value,
            4 => {
                self.oam[self.oam_addr as usize] = value;
                self.oam_addr = self.oam_addr.wrapping_add(1);
            }
            5 => {
                if self.first_write {
                    self.fine_x = value & 0x07;
                    self.scroll_x_low = value;
                    self.update_scroll_x_px();
                    self.temp_addr = (self.temp_addr & 0xffe0) | ((value as u16) >> 3);
                } else {
                    self.scroll_y_px = value as u16;
                    self.temp_addr = (self.temp_addr & 0x8fff) | (((value as u16) & 0x07) << 12);
                    self.temp_addr = (self.temp_addr & 0xfc1f) | (((value as u16) & 0xf8) << 2);
                }
                self.scroll_addr = self.temp_addr;
                self.first_write = !self.first_write;
            }
            6 => {
                if self.first_write {
                    self.temp_addr = (self.temp_addr & 0x00ff) | (((value as u16) & 0x3f) << 8);
                } else {
                    self.temp_addr = (self.temp_addr & 0xff00) | value as u16;
                    self.addr = self.temp_addr;
                }
                self.first_write = !self.first_write;
            }
            7 => self.write_data(value),
            _ => {}
        }
    }

    #[inline]
    fn read_data(&mut self) -> u8 {
        let addr = self.addr & 0x3fff;
        let inc = if self.ctrl & 0x04 != 0 { 32 } else { 1 };
        self.addr = self.addr.wrapping_add(inc) & 0x3fff;

        if addr >= 0x3f00 {
            self.ppu_read(addr)
        } else {
            let buffered = self.data_buffer;
            self.data_buffer = self.ppu_read(addr);
            buffered
        }
    }

    #[inline]
    fn write_data(&mut self, value: u8) {
        let addr = self.addr & 0x3fff;
        self.ppu_write(addr, value);
        let inc = if self.ctrl & 0x04 != 0 { 32 } else { 1 };
        self.addr = self.addr.wrapping_add(inc) & 0x3fff;
    }

    #[inline]
    fn chr_read(&self, addr: usize) -> u8 {
        let idx = addr & self.chr_addr_mask;
        // SAFETY: SMB/NROM CHR ROM sizes are power-of-two and chr_addr_mask is len - 1.
        unsafe { *self.chr_rom.get_unchecked(idx) }
    }

    #[inline]
    fn ppu_read(&self, addr: u16) -> u8 {
        let addr = addr & 0x3fff;
        match addr {
            0x0000..=0x1fff => self.chr_read(addr as usize),
            0x2000..=0x3eff => {
                let idx = self.mirror_nametable_addr(addr);
                self.vram[idx]
            }
            0x3f00..=0x3fff => self.palette[self.mirror_palette_addr(addr)],
            _ => 0,
        }
    }

    #[inline(always)]
    fn nametable_read(&self, table: usize, offset: usize) -> u8 {
        let physical_table = if self.vertical_mirroring {
            table & 1
        } else {
            (table >> 1) & 1
        };
        // SAFETY: mirroring maps the four logical nametables into the two
        // physical 1 KiB VRAM pages, and the offset is masked to that page.
        unsafe {
            *self
                .vram
                .get_unchecked(physical_table * 0x400 + (offset & 0x3ff))
        }
    }

    #[inline]
    fn ppu_write(&mut self, addr: u16, value: u8) {
        let addr = addr & 0x3fff;
        match addr {
            0x0000..=0x1fff => {}
            0x2000..=0x3eff => {
                let idx = self.mirror_nametable_addr(addr);
                self.vram[idx] = value;
            }
            0x3f00..=0x3fff => {
                let idx = self.mirror_palette_addr(addr);
                self.palette[idx] = value;
            }
            _ => {}
        }
    }

    #[inline]
    fn mirror_nametable_addr(&self, addr: u16) -> usize {
        let v = (addr - 0x2000) as usize % 0x1000;
        let table = v / 0x400;
        let offset = v & 0x3ff;
        let physical_table = if self.vertical_mirroring {
            table & 1
        } else {
            (table >> 1) & 1
        };
        physical_table * 0x400 + offset
    }

    #[inline]
    fn mirror_palette_addr(&self, addr: u16) -> usize {
        let mut idx = (addr as usize - 0x3f00) & 0x1f;
        if idx == 0x10 {
            idx = 0x00;
        } else if idx == 0x14 {
            idx = 0x04;
        } else if idx == 0x18 {
            idx = 0x08;
        } else if idx == 0x1c {
            idx = 0x0c;
        }
        idx
    }

    #[allow(dead_code)]
    fn write_gray_frame(&self, dst: &mut [u8]) {
        debug_assert_eq!(dst.len(), FRAME_PIXELS_GRAY);
        for y in 0..NES_HEIGHT {
            for x in 0..NES_WIDTH {
                let color = self.bg_color_index(x, y);
                dst[y * NES_WIDTH + x] = NES_GRAY_PALETTE[color as usize];
            }
        }
        self.draw_sprites_gray(dst);
    }

    #[allow(dead_code)]
    fn write_gray_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        debug_assert_eq!(dst.len(), NES_WIDTH * height);
        self.write_bg_gray_cropped_tiled(dst, crop_top, height);
        self.draw_sprites_gray_cropped(dst, crop_top, height);
    }

    fn write_gray_frame_region(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        crop_left: usize,
        width: usize,
        height: usize,
    ) {
        debug_assert_eq!(dst.len(), width * height);
        self.write_bg_gray_region_tiled(dst, crop_top, crop_left, width, height);
        self.draw_sprites_gray_region(dst, crop_top, crop_left, width, height);
    }

    fn write_gray_frame_cropped_area_84x84(&self, dst: &mut [u8], sprite_shadow: &mut [u8]) {
        debug_assert_eq!(dst.len(), DEFAULT_GRAY_RESIZE_PIXELS);
        let native_len = VISIBLE_FRAME_WIDTH * DEFAULT_GRAY_CROP_HEIGHT;
        debug_assert!(sprite_shadow.len() >= native_len);
        let native = &mut sprite_shadow[..native_len];
        self.write_gray_frame_region(
            native,
            VISIBLE_FRAME_TOP + DEFAULT_GRAY_CROP_TOP,
            VISIBLE_FRAME_LEFT,
            VISIBLE_FRAME_WIDTH,
            DEFAULT_GRAY_CROP_HEIGHT,
        );

        for dy in 0..DEFAULT_GRAY_RESIZE_HEIGHT {
            let y0 = (dy * DEFAULT_GRAY_CROP_HEIGHT) / DEFAULT_GRAY_RESIZE_HEIGHT;
            let y1 = (((dy + 1) * DEFAULT_GRAY_CROP_HEIGHT) / DEFAULT_GRAY_RESIZE_HEIGHT)
                .max(y0 + 1)
                .min(DEFAULT_GRAY_CROP_HEIGHT);
            for dx in 0..DEFAULT_GRAY_RESIZE_WIDTH {
                let x0 = (dx * VISIBLE_FRAME_WIDTH) / DEFAULT_GRAY_RESIZE_WIDTH;
                let x1 = (((dx + 1) * VISIBLE_FRAME_WIDTH) / DEFAULT_GRAY_RESIZE_WIDTH)
                    .max(x0 + 1)
                    .min(VISIBLE_FRAME_WIDTH);
                let mut sum = 0u32;
                for sy in y0..y1 {
                    let src_row = sy * VISIBLE_FRAME_WIDTH;
                    for sx in x0..x1 {
                        sum += native[src_row + sx] as u32;
                    }
                }
                dst[dy * DEFAULT_GRAY_RESIZE_WIDTH + dx] =
                    (sum / ((x1 - x0) * (y1 - y0)) as u32) as u8;
            }
        }
    }

    #[allow(dead_code)]
    fn write_bg_gray_cropped_tiled(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        let palette_gray = self.palette_gray();
        if self.mask & 0x08 == 0 {
            dst.fill(palette_gray[0]);
            return;
        }

        let pattern_base = if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        };
        let scroll_x = self.render_scroll_x_px() as usize;
        let scroll_y = self.scroll_y_px as usize;

        for out_y in 0..height {
            let y = crop_top + out_y;
            let world_y = if y < 32 { y } else { y + scroll_y };
            let table_y = (world_y / 240) & 1;
            let local_y = world_y % 240;
            let tile_y = local_y / 8;
            let fine_y = local_y & 7;
            let row_start = out_y * NES_WIDTH;
            let mut x = 0usize;

            while x < NES_WIDTH {
                let world_x = if y < 32 { x } else { x + scroll_x };
                let table_x = (world_x / 256) & 1;
                let table = table_y * 2 + table_x;
                let local_x = world_x & 0xff;
                let tile_x = local_x / 8;
                let fine_x = local_x & 7;
                let nt_base = 0x2000 + (table as u16) * 0x400;
                let tile_id = self.ppu_read(nt_base + (tile_y * 32 + tile_x) as u16) as usize;
                let attr =
                    self.ppu_read(nt_base + 0x3c0 + ((tile_y / 4) * 8 + (tile_x / 4)) as u16);
                let shift = ((tile_y & 0x02) << 1) | (tile_x & 0x02);
                let palette_id = (attr >> shift) & 0x03;
                let pattern_addr = pattern_base + tile_id * 16 + fine_y;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                let run = (8 - fine_x).min(NES_WIDTH - x);

                for col in 0..run {
                    let bit = 7 - (fine_x + col);
                    let pixel = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1);
                    let gray = if pixel == 0 {
                        palette_gray[0]
                    } else {
                        palette_gray[(palette_id as usize) * 4 + pixel as usize]
                    };
                    dst[row_start + x + col] = gray;
                }

                x += run;
            }
        }
    }

    fn write_bg_gray_region_tiled(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        crop_left: usize,
        width: usize,
        height: usize,
    ) {
        let palette_gray = self.palette_gray();
        if self.mask & 0x08 == 0 {
            dst.fill(palette_gray[0]);
            return;
        }

        let pattern_base = if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        };
        let scroll_x = self.render_scroll_x_px() as usize;
        let scroll_y = self.scroll_y_px as usize;

        for out_y in 0..height {
            let y = crop_top + out_y;
            let world_y = if y < 32 { y } else { y + scroll_y };
            let table_y = (world_y / 240) & 1;
            let local_y = world_y % 240;
            let tile_y = local_y / 8;
            let fine_y = local_y & 7;
            let row_start = out_y * width;
            let mut out_x = 0usize;

            while out_x < width {
                let screen_x = crop_left + out_x;
                let world_x = if y < 32 {
                    screen_x
                } else {
                    screen_x + scroll_x
                };
                let table_x = (world_x / 256) & 1;
                let table = table_y * 2 + table_x;
                let local_x = world_x & 0xff;
                let tile_x = local_x / 8;
                let fine_x = local_x & 7;
                let nt_base = 0x2000 + (table as u16) * 0x400;
                let tile_id = self.ppu_read(nt_base + (tile_y * 32 + tile_x) as u16) as usize;
                let attr =
                    self.ppu_read(nt_base + 0x3c0 + ((tile_y / 4) * 8 + (tile_x / 4)) as u16);
                let shift = ((tile_y & 0x02) << 1) | (tile_x & 0x02);
                let palette_id = (attr >> shift) & 0x03;
                let pattern_addr = pattern_base + tile_id * 16 + fine_y;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                let run = (8 - fine_x).min(width - out_x);

                for col in 0..run {
                    let bit = 7 - (fine_x + col);
                    let pixel = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1);
                    let gray = if pixel == 0 {
                        palette_gray[0]
                    } else {
                        palette_gray[(palette_id as usize) * 4 + pixel as usize]
                    };
                    dst[row_start + out_x + col] = gray;
                }

                out_x += run;
            }
        }
    }

    #[allow(dead_code)]
    fn write_rgb_frame(&self, dst: &mut [u8]) {
        debug_assert_eq!(dst.len(), FRAME_PIXELS_RGB);
        let plane = NES_WIDTH * NES_HEIGHT;
        for y in 0..NES_HEIGHT {
            for x in 0..NES_WIDTH {
                let color = nes_rgb(self.bg_color_index(x, y));
                let idx = y * NES_WIDTH + x;
                dst[idx] = color[0];
                dst[plane + idx] = color[1];
                dst[plane * 2 + idx] = color[2];
            }
        }
        self.draw_sprites_rgb(dst);
    }

    #[allow(dead_code)]
    fn write_rgb_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        debug_assert_eq!(dst.len(), NES_WIDTH * height * RGB_CHANNELS);
        let plane = NES_WIDTH * height;
        for out_y in 0..height {
            let y = crop_top + out_y;
            for x in 0..NES_WIDTH {
                let color = nes_rgb(self.bg_color_index(x, y));
                let idx = out_y * NES_WIDTH + x;
                dst[idx] = color[0];
                dst[plane + idx] = color[1];
                dst[plane * 2 + idx] = color[2];
            }
        }
        self.draw_sprites_rgb_cropped(dst, crop_top, height);
    }

    fn write_rgb_frame_region(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        crop_left: usize,
        width: usize,
        height: usize,
    ) {
        debug_assert_eq!(dst.len(), width * height * RGB_CHANNELS);
        let plane = width * height;
        for out_y in 0..height {
            let y = crop_top + out_y;
            for out_x in 0..width {
                let x = crop_left + out_x;
                let color = nes_rgb(self.bg_color_index(x, y));
                let idx = out_y * width + out_x;
                dst[idx] = color[0];
                dst[plane + idx] = color[1];
                dst[plane * 2 + idx] = color[2];
            }
        }
        self.draw_sprites_rgb_region(dst, crop_top, crop_left, width, height);
    }

    #[inline]
    fn bg_color_index(&self, x: usize, y: usize) -> u8 {
        if self.mask & 0x08 == 0 {
            return self.palette[0];
        }

        let (world_x, world_y) = self.bg_world_pos(x, y);
        let table_x = (world_x / 256) & 1;
        let table_y = (world_y / 240) & 1;
        let table = table_y * 2 + table_x;

        let local_x = world_x & 0xff;
        let local_y = world_y % 240;
        let tile_x = local_x / 8;
        let tile_y = local_y / 8;
        let fine_x = local_x & 7;
        let fine_y = local_y & 7;

        let nt_base = 0x2000 + (table as u16) * 0x400;
        let tile_id = self.ppu_read(nt_base + (tile_y * 32 + tile_x) as u16) as usize;
        let attr = self.ppu_read(nt_base + 0x3c0 + ((tile_y / 4) * 8 + (tile_x / 4)) as u16);
        let shift = ((tile_y & 0x02) << 1) | (tile_x & 0x02);
        let palette_id = (attr >> shift) & 0x03;

        let pattern_base = if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        };
        let pattern_addr = pattern_base + tile_id * 16 + fine_y;
        let lo = self.chr_read(pattern_addr);
        let hi = self.chr_read(pattern_addr + 8);
        let bit = 7 - fine_x;
        let pixel = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1);
        if pixel == 0 {
            self.palette[0]
        } else {
            self.palette[(palette_id as usize) * 4 + pixel as usize]
        }
    }

    #[inline]
    fn bg_pixel_opaque(&self, x: usize, y: usize) -> bool {
        if self.mask & 0x08 == 0 {
            return false;
        }

        let (world_x, world_y) = self.bg_world_pos(x, y);
        let table_x = (world_x / 256) & 1;
        let table_y = (world_y / 240) & 1;
        let table = table_y * 2 + table_x;

        let local_x = world_x & 0xff;
        let local_y = world_y % 240;
        let tile_x = local_x / 8;
        let tile_y = local_y / 8;
        let fine_x = local_x & 7;
        let fine_y = local_y & 7;

        let nt_base = 0x2000 + (table as u16) * 0x400;
        let tile_id = self.ppu_read(nt_base + (tile_y * 32 + tile_x) as u16) as usize;
        let pattern_base = if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        };
        let pattern_addr = pattern_base + tile_id * 16 + fine_y;
        let lo = self.chr_read(pattern_addr);
        let hi = self.chr_read(pattern_addr + 8);
        let bit = 7 - fine_x;
        (((lo >> bit) & 1) | (((hi >> bit) & 1) << 1)) != 0
    }

    fn sprite_scanline_mask(&self) -> [u64; NES_HEIGHT] {
        let mut mask = [0u64; NES_HEIGHT];
        let mut counts = [0u8; NES_HEIGHT];
        for sprite in 0..64usize {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if !(0..NES_HEIGHT as i16).contains(&screen_y) {
                    continue;
                }
                let screen_y = screen_y as usize;
                if counts[screen_y] >= 8 {
                    continue;
                }
                counts[screen_y] += 1;
                mask[screen_y] |= 1u64 << sprite;
            }
        }
        mask
    }

    fn draw_sprites_gray(&self, dst: &mut [u8]) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let palette_gray = self.palette_gray();
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        let sprite_scanline_mask = self.sprite_scanline_mask();
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if !(0..NES_HEIGHT as i16).contains(&screen_y) {
                    continue;
                }
                if sprite_scanline_mask[screen_y as usize] & (1u64 << sprite) == 0 {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if !(0..NES_WIDTH as i16).contains(&screen_x) {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    let idx = screen_y as usize * NES_WIDTH + screen_x as usize;
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        dst[idx] = NES_GRAY_PALETTE
                            [self.bg_color_index(screen_x as usize, screen_y as usize) as usize];
                        continue;
                    }
                    dst[idx] = palette_gray[palette_base + pixel as usize];
                }
            }
        }
    }

    #[allow(dead_code)]
    fn draw_sprites_gray_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let palette_gray = self.palette_gray();
        let crop_top = crop_top as i16;
        let crop_bottom = crop_top + height as i16;
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        let sprite_scanline_mask = self.sprite_scanline_mask();
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if screen_y < crop_top || screen_y >= crop_bottom {
                    continue;
                }
                if sprite_scanline_mask[screen_y as usize] & (1u64 << sprite) == 0 {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if !(0..NES_WIDTH as i16).contains(&screen_x) {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    let out_y = (screen_y - crop_top) as usize;
                    let idx = out_y * NES_WIDTH + screen_x as usize;
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        dst[idx] = NES_GRAY_PALETTE
                            [self.bg_color_index(screen_x as usize, screen_y as usize) as usize];
                        continue;
                    }
                    dst[idx] = palette_gray[palette_base + pixel as usize];
                }
            }
        }
    }

    fn draw_sprites_gray_region(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        crop_left: usize,
        width: usize,
        height: usize,
    ) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let palette_gray = self.palette_gray();
        let crop_top_i = crop_top as i16;
        let crop_bottom = crop_top_i + height as i16;
        let crop_left_i = crop_left as i16;
        let crop_right = crop_left_i + width as i16;
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        let sprite_scanline_mask = self.sprite_scanline_mask();
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if screen_y < crop_top_i || screen_y >= crop_bottom {
                    continue;
                }
                if sprite_scanline_mask[screen_y as usize] & (1u64 << sprite) == 0 {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if screen_x < crop_left_i || screen_x >= crop_right {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    let out_y = (screen_y - crop_top_i) as usize;
                    let out_x = (screen_x - crop_left_i) as usize;
                    let idx = out_y * width + out_x;
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        dst[idx] = NES_GRAY_PALETTE
                            [self.bg_color_index(screen_x as usize, screen_y as usize) as usize];
                        continue;
                    }
                    dst[idx] = palette_gray[palette_base + pixel as usize];
                }
            }
        }
    }

    #[allow(dead_code)]
    fn draw_sprites_rgb(&self, dst: &mut [u8]) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let plane = NES_WIDTH * NES_HEIGHT;
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        let sprite_scanline_mask = self.sprite_scanline_mask();
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if !(0..NES_HEIGHT as i16).contains(&screen_y) {
                    continue;
                }
                if sprite_scanline_mask[screen_y as usize] & (1u64 << sprite) == 0 {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if !(0..NES_WIDTH as i16).contains(&screen_x) {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    let idx = screen_y as usize * NES_WIDTH + screen_x as usize;
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        let color =
                            nes_rgb(self.bg_color_index(screen_x as usize, screen_y as usize));
                        dst[idx] = color[0];
                        dst[plane + idx] = color[1];
                        dst[plane * 2 + idx] = color[2];
                        continue;
                    }
                    let color = nes_rgb(self.palette[palette_base + pixel as usize]);
                    dst[idx] = color[0];
                    dst[plane + idx] = color[1];
                    dst[plane * 2 + idx] = color[2];
                }
            }
        }
    }

    #[allow(dead_code)]
    fn draw_sprites_rgb_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let crop_top = crop_top as i16;
        let crop_bottom = crop_top + height as i16;
        let plane = NES_WIDTH * height;
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        let sprite_scanline_mask = self.sprite_scanline_mask();
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if screen_y < crop_top || screen_y >= crop_bottom {
                    continue;
                }
                if sprite_scanline_mask[screen_y as usize] & (1u64 << sprite) == 0 {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if !(0..NES_WIDTH as i16).contains(&screen_x) {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    let out_y = (screen_y - crop_top) as usize;
                    let idx = out_y * NES_WIDTH + screen_x as usize;
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        let color =
                            nes_rgb(self.bg_color_index(screen_x as usize, screen_y as usize));
                        dst[idx] = color[0];
                        dst[plane + idx] = color[1];
                        dst[plane * 2 + idx] = color[2];
                        continue;
                    }
                    let color = nes_rgb(self.palette[palette_base + pixel as usize]);
                    dst[idx] = color[0];
                    dst[plane + idx] = color[1];
                    dst[plane * 2 + idx] = color[2];
                }
            }
        }
    }

    fn draw_sprites_rgb_region(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        crop_left: usize,
        width: usize,
        height: usize,
    ) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let crop_top_i = crop_top as i16;
        let crop_bottom = crop_top_i + height as i16;
        let crop_left_i = crop_left as i16;
        let crop_right = crop_left_i + width as i16;
        let plane = width * height;
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        let sprite_scanline_mask = self.sprite_scanline_mask();
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if screen_y < crop_top_i || screen_y >= crop_bottom {
                    continue;
                }
                if sprite_scanline_mask[screen_y as usize] & (1u64 << sprite) == 0 {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if screen_x < crop_left_i || screen_x >= crop_right {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    let out_y = (screen_y - crop_top_i) as usize;
                    let out_x = (screen_x - crop_left_i) as usize;
                    let idx = out_y * width + out_x;
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        let color =
                            nes_rgb(self.bg_color_index(screen_x as usize, screen_y as usize));
                        dst[idx] = color[0];
                        dst[plane + idx] = color[1];
                        dst[plane * 2 + idx] = color[2];
                        continue;
                    }
                    let color = nes_rgb(self.palette[palette_base + pixel as usize]);
                    dst[idx] = color[0];
                    dst[plane + idx] = color[1];
                    dst[plane * 2 + idx] = color[2];
                }
            }
        }
    }

    #[inline]
    fn update_scroll_x_px(&mut self) {
        self.scroll_x_px = (((self.ctrl & 0x01) as u16) << 8) | self.scroll_x_low as u16;
    }

    #[inline]
    fn set_scroll_override_x(&mut self, scroll_x_px: Option<u16>) {
        self.scroll_override_x_px = scroll_x_px;
    }

    #[inline]
    fn render_scroll_x_px(&self) -> u16 {
        self.scroll_override_x_px.unwrap_or(self.scroll_x_px)
    }

    #[inline]
    fn palette_gray(&self) -> [u8; 32] {
        let mut out = [0; 32];
        for (dst, &color) in out.iter_mut().zip(self.palette.iter()) {
            *dst = NES_GRAY_PALETTE[color as usize];
        }
        out
    }

    #[inline]
    fn bg_world_pos(&self, x: usize, y: usize) -> (usize, usize) {
        if y < 32 {
            (x, y)
        } else {
            (
                x + self.render_scroll_x_px() as usize,
                y + self.scroll_y_px as usize,
            )
        }
    }
}

const NES_GRAY_PALETTE: [u8; 64] = build_nes_gray_palette();

const fn build_nes_gray_palette() -> [u8; 64] {
    let mut table = [0; 64];
    let mut color = 0;
    while color < 64 {
        let rgb = NES_RGB_PALETTE[color];
        table[color] = (((rgb[0] as u32) * 77 + (rgb[1] as u32) * 150 + (rgb[2] as u32) * 29 + 128)
            >> 8) as u8;
        color += 1;
    }
    table
}

#[inline]
fn nes_rgb(color: u8) -> [u8; 3] {
    NES_RGB_PALETTE[(color as usize) & 0x3f]
}

const NES_RGB_PALETTE: [[u8; 3]; 64] = [
    [112, 116, 112],
    [32, 24, 136],
    [0, 0, 168],
    [64, 0, 152],
    [136, 0, 112],
    [168, 0, 16],
    [160, 0, 0],
    [120, 8, 0],
    [64, 44, 0],
    [0, 68, 0],
    [0, 80, 0],
    [0, 60, 16],
    [24, 60, 88],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [184, 188, 184],
    [0, 112, 232],
    [32, 56, 232],
    [128, 0, 240],
    [184, 0, 184],
    [224, 0, 88],
    [216, 40, 0],
    [200, 76, 8],
    [136, 112, 0],
    [0, 148, 0],
    [0, 168, 0],
    [0, 144, 56],
    [0, 128, 136],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [248, 252, 248],
    [56, 188, 248],
    [88, 148, 248],
    [64, 136, 248],
    [240, 120, 248],
    [248, 116, 176],
    [248, 116, 96],
    [248, 152, 56],
    [240, 188, 56],
    [128, 208, 16],
    [72, 220, 72],
    [88, 248, 152],
    [0, 232, 216],
    [120, 120, 120],
    [0, 0, 0],
    [0, 0, 0],
    [248, 252, 248],
    [168, 228, 248],
    [192, 212, 248],
    [208, 200, 248],
    [248, 196, 248],
    [248, 196, 216],
    [248, 188, 176],
    [248, 216, 168],
    [248, 228, 160],
    [224, 252, 160],
    [168, 240, 184],
    [176, 252, 200],
    [152, 252, 240],
    [192, 196, 192],
    [0, 0, 0],
    [0, 0, 0],
];

#[inline]
fn next_ppu_event_dot(current: usize) -> usize {
    if current < PPU_VBLANK_DOT {
        PPU_VBLANK_DOT
    } else if current < PPU_PRERENDER_DOT {
        PPU_PRERENDER_DOT
    } else if current < PPU_SPRITE0_DOT {
        PPU_SPRITE0_DOT
    } else {
        PPU_DOTS_PER_FRAME
    }
}

fn required_field<'a>(
    state: &'a [u8],
    name: &'static [u8; 4],
    display_name: &'static str,
    size: usize,
) -> Result<&'a [u8], StateLoadError> {
    optional_field(state, name, size).ok_or(StateLoadError::MissingField {
        name: display_name,
        size,
    })
}

fn optional_field<'a>(state: &'a [u8], name: &[u8; 4], size: usize) -> Option<&'a [u8]> {
    let header_len = name.len() + 4;
    state
        .windows(header_len)
        .enumerate()
        .find_map(|(offset, window)| {
            if &window[..4] != name {
                return None;
            }
            let field_size = u32::from_le_bytes(window[4..8].try_into().ok()?) as usize;
            if field_size != size {
                return None;
            }
            let start = offset + header_len;
            let end = start.checked_add(field_size)?;
            state.get(start..end)
        })
}

fn read_u16_le(value: &[u8]) -> Option<u16> {
    Some(u16::from_le_bytes(value.get(..2)?.try_into().ok()?))
}

#[inline]
fn sign_extend_u8(value: u8) -> i16 {
    (value as i8) as i16
}

#[inline]
fn decode_ln_bcd_be(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .fold(0u32, |value, byte| value * 10 + u32::from(byte & 0x0f))
}

#[derive(Clone)]
pub struct NesEmulator {
    cpu: Cpu,
    ppu: Ppu,
    ram: [u8; 2048],
    prg_rom: Vec<u8>,
    prg_addr_mask: usize,
    controller_state: u8,
    controller_shift: u8,
    controller_strobe: bool,
    extra_cycles: u16,
    x_pos: u16,
    coins: u8,
    level_hi: i16,
    level_lo: i16,
    lives: i16,
    score: u32,
    scrolling: i16,
    time: u16,
    xscroll_hi: u8,
    xscroll_lo: u8,
    terminate_on_flag: bool,
    done: bool,
}

impl NesEmulator {
    pub fn new_with_options(cart: Cartridge, terminate_on_flag: bool) -> Self {
        let prg_addr_mask = cart.prg_rom.len() - 1;
        let ppu = Ppu::new(cart.chr_rom, cart.vertical_mirroring);
        let mut emu = Self {
            cpu: Cpu::new(),
            ppu,
            ram: [0; 2048],
            prg_rom: cart.prg_rom,
            prg_addr_mask,
            controller_state: 0,
            controller_shift: 0,
            controller_strobe: false,
            extra_cycles: 0,
            x_pos: 0,
            coins: 0,
            level_hi: 0,
            level_lo: 0,
            lives: 0,
            score: 0,
            scrolling: 0,
            time: 0,
            xscroll_hi: 0,
            xscroll_lo: 0,
            terminate_on_flag,
            done: false,
        };
        emu.reset();
        emu
    }

    pub fn reset(&mut self) {
        self.cpu = Cpu::new();
        self.ppu.reset();
        self.ram = [0; 2048];
        self.controller_state = 0;
        self.controller_shift = 0;
        self.controller_strobe = false;
        self.extra_cycles = 0;
        self.done = false;
        self.cpu.pc = self.cpu_read_u16(0xfffc);
        self.refresh_smb_state();
    }

    pub fn load_fceu_state(&mut self, state: &[u8]) -> Result<(), StateLoadError> {
        let pc = required_field(state, b"PC\0\0", "PC", 2)?;
        let a = required_field(state, b"A\0\0\0", "A", 1)?[0];
        let x = required_field(state, b"X\0\0\0", "X", 1)?[0];
        let y = required_field(state, b"Y\0\0\0", "Y", 1)?[0];
        let sp = required_field(state, b"S\0\0\0", "S", 1)?[0];
        let p = required_field(state, b"P\0\0\0", "P", 1)?[0];
        let ram = required_field(state, b"RAM\0", "RAM", 2048)?;
        let ntar = required_field(state, b"NTAR", "NTAR", 2048)?;
        let pram = required_field(state, b"PRAM", "PRAM", 32)?;
        let spra = required_field(state, b"SPRA", "SPRA", 256)?;
        let ppur = required_field(state, b"PPUR", "PPUR", 4)?;

        self.cpu = Cpu {
            a,
            x,
            y,
            sp,
            pc: read_u16_le(pc).unwrap_or(0),
            p,
        };
        self.ppu.load_fceu_state(
            ntar,
            pram,
            spra,
            ppur,
            optional_field(state, b"RADD", 2),
            optional_field(state, b"TADD", 2),
            optional_field(state, b"XOFF", 1),
        );
        self.ram.copy_from_slice(ram);
        self.controller_state = 0;
        self.controller_shift = 0;
        self.controller_strobe = false;
        self.extra_cycles = 0;
        self.done = false;
        self.refresh_smb_state();
        // FCEU state files resume close to, but not exactly at, this simplified
        // PPU frame boundary. One no-op frame matches stable-retro's first
        // visible gameplay frame without changing the reset observation.
        self.run_frame(0);
        self.refresh_smb_state();
        Ok(())
    }

    #[inline]
    pub fn step_frame(&mut self, action: MarioAction) -> f32 {
        if self.done {
            return 0.0;
        }

        let before = self.xscroll_lo;
        self.run_frame(action.buttons());
        self.refresh_smb_state();
        if self.terminate_on_flag && self.x_pos >= 3160 {
            self.done = true;
        }
        (self.xscroll_lo as i16 - before as i16).max(0) as f32
    }

    #[inline]
    #[allow(dead_code)]
    pub fn write_rgb_frame(&self, dst: &mut [u8]) {
        self.ppu.write_rgb_frame(dst);
    }

    #[inline]
    #[allow(dead_code)]
    pub fn write_rgb_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        self.ppu.write_rgb_frame_cropped(dst, crop_top, height);
    }

    #[inline]
    pub fn write_rgb_visible_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        self.ppu.write_rgb_frame_region(
            dst,
            VISIBLE_FRAME_TOP + crop_top,
            VISIBLE_FRAME_LEFT,
            VISIBLE_FRAME_WIDTH,
            height,
        );
    }

    #[inline]
    #[allow(dead_code)]
    pub fn write_gray_frame(&self, dst: &mut [u8]) {
        self.ppu.write_gray_frame(dst);
    }

    #[inline]
    #[allow(dead_code)]
    pub fn write_gray_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        self.ppu.write_gray_frame_cropped(dst, crop_top, height);
    }

    #[inline]
    pub fn write_gray_visible_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        self.ppu.write_gray_frame_region(
            dst,
            VISIBLE_FRAME_TOP + crop_top,
            VISIBLE_FRAME_LEFT,
            VISIBLE_FRAME_WIDTH,
            height,
        );
    }

    #[inline]
    pub fn write_gray_frame_cropped_area_84x84(&self, dst: &mut [u8], sprite_shadow: &mut [u8]) {
        self.ppu
            .write_gray_frame_cropped_area_84x84(dst, sprite_shadow);
    }

    #[inline]
    pub fn x_pos(&self) -> u16 {
        self.x_pos
    }

    #[inline]
    pub fn coins(&self) -> u8 {
        self.coins
    }

    #[inline]
    pub fn level_hi(&self) -> i16 {
        self.level_hi
    }

    #[inline]
    pub fn level_lo(&self) -> i16 {
        self.level_lo
    }

    #[inline]
    pub fn lives(&self) -> i16 {
        self.lives
    }

    #[inline]
    pub fn score(&self) -> u32 {
        self.score
    }

    #[inline]
    pub fn scrolling(&self) -> i16 {
        self.scrolling
    }

    #[inline]
    pub fn time(&self) -> u16 {
        self.time
    }

    #[inline]
    pub fn xscroll_hi(&self) -> u8 {
        self.xscroll_hi
    }

    #[inline]
    pub fn xscroll_lo(&self) -> u8 {
        self.xscroll_lo
    }

    #[inline]
    pub fn ram(&self) -> &[u8; 2048] {
        &self.ram
    }

    #[inline]
    pub fn oam(&self) -> &[u8; 256] {
        self.ppu.oam()
    }

    #[inline]
    pub fn debug_bg_pixel(&self, x: usize, y: usize) -> (u8, bool) {
        self.ppu.debug_bg_pixel(x, y)
    }

    #[inline]
    pub fn is_done(&self) -> bool {
        self.done
    }

    fn run_frame(&mut self, buttons: u8) {
        self.controller_state = buttons;
        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        loop {
            if self.ppu.take_nmi() {
                self.interrupt(0xfffa, false);
            }
            let cycles = self.cpu_step() as usize;
            cpu_cycle_guard += cycles;
            pending_ppu_cycles += cycles * 3;
            let must_flush_ppu = pending_ppu_cycles >= self.ppu.cycles_until_next_event()
                || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD;
            if must_flush_ppu {
                if self.ppu.tick(pending_ppu_cycles)
                    || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
                {
                    pending_ppu_cycles = 0;
                    break;
                }
                pending_ppu_cycles = 0;
            }
        }
        if pending_ppu_cycles > 0 {
            self.ppu.tick(pending_ppu_cycles);
        }
    }

    #[inline]
    fn refresh_smb_state(&mut self) {
        self.x_pos = ((self.ram[0x006d] as u16) << 8) | self.ram[0x0086] as u16;
        self.coins = self.ram[0x075e];
        self.level_hi = sign_extend_u8(self.ram[0x075f]);
        self.level_lo = sign_extend_u8(self.ram[0x075c]);
        self.lives = sign_extend_u8(self.ram[0x075a]);
        self.score = decode_ln_bcd_be(&self.ram[0x07dd..0x07e3]) as u32;
        self.scrolling = sign_extend_u8(self.ram[0x0778]);
        self.time = decode_ln_bcd_be(&self.ram[0x07f8..0x07fb]) as u16;
        self.xscroll_hi = self.ram[0x071a];
        self.xscroll_lo = self.ram[0x071c];
        self.ppu.set_scroll_override_x(None);
    }

    #[inline]
    fn cpu_read(&mut self, addr: u16) -> u8 {
        match addr {
            0x0000..=0x1fff => self.ram[addr as usize & 0x07ff],
            0x2000..=0x3fff => self.ppu.cpu_read_register(addr),
            0x4016 => self.controller_read(),
            0x8000..=0xffff => self.prg_read(addr),
            _ => 0,
        }
    }

    #[inline]
    fn cpu_write(&mut self, addr: u16, value: u8) {
        match addr {
            0x0000..=0x1fff => self.ram[addr as usize & 0x07ff] = value,
            0x2000..=0x3fff => self.ppu.cpu_write_register(addr, value),
            0x4014 => self.oam_dma(value),
            0x4016 => self.controller_write(value),
            _ => {}
        }
    }

    #[inline]
    fn prg_read(&self, addr: u16) -> u8 {
        let idx = ((addr - 0x8000) as usize) & self.prg_addr_mask;
        // SAFETY: SMB/NROM PRG ROM sizes are power-of-two and prg_addr_mask is len - 1.
        unsafe { *self.prg_rom.get_unchecked(idx) }
    }

    #[inline]
    fn cpu_read_u16(&mut self, addr: u16) -> u16 {
        let lo = self.cpu_read(addr) as u16;
        let hi = self.cpu_read(addr.wrapping_add(1)) as u16;
        lo | (hi << 8)
    }

    #[inline]
    fn controller_write(&mut self, value: u8) {
        self.controller_strobe = value & 1 != 0;
        if self.controller_strobe {
            self.controller_shift = self.controller_state;
        }
    }

    #[inline]
    fn controller_read(&mut self) -> u8 {
        if self.controller_strobe {
            return 0x40 | (self.controller_state & 1);
        }
        let value = self.controller_shift & 1;
        self.controller_shift = (self.controller_shift >> 1) | 0x80;
        0x40 | value
    }

    fn oam_dma(&mut self, page: u8) {
        let base = (page as u16) << 8;
        for i in 0..256u16 {
            let value = self.cpu_read(base | i);
            let idx = self.ppu.oam_addr.wrapping_add(i as u8) as usize;
            self.ppu.oam[idx] = value;
        }
        self.extra_cycles = self.extra_cycles.wrapping_add(513);
    }

    #[inline]
    fn fetch_u8(&mut self) -> u8 {
        let value = if self.cpu.pc >= 0x8000 {
            self.prg_read(self.cpu.pc)
        } else {
            self.cpu_read(self.cpu.pc)
        };
        self.cpu.pc = self.cpu.pc.wrapping_add(1);
        value
    }

    #[inline]
    fn fetch_u16(&mut self) -> u16 {
        let lo = self.fetch_u8() as u16;
        let hi = self.fetch_u8() as u16;
        lo | (hi << 8)
    }

    #[inline]
    fn zp(&mut self) -> u16 {
        self.fetch_u8() as u16
    }

    #[inline]
    fn zpx(&mut self) -> u16 {
        self.fetch_u8().wrapping_add(self.cpu.x) as u16
    }

    #[inline]
    fn zpy(&mut self) -> u16 {
        self.fetch_u8().wrapping_add(self.cpu.y) as u16
    }

    #[inline]
    fn abs(&mut self) -> u16 {
        self.fetch_u16()
    }

    #[inline]
    fn absx(&mut self) -> (u16, bool) {
        let base = self.fetch_u16();
        let addr = base.wrapping_add(self.cpu.x as u16);
        (addr, page_crossed(base, addr))
    }

    #[inline]
    fn absy(&mut self) -> (u16, bool) {
        let base = self.fetch_u16();
        let addr = base.wrapping_add(self.cpu.y as u16);
        (addr, page_crossed(base, addr))
    }

    #[inline]
    fn indx(&mut self) -> u16 {
        let ptr = self.fetch_u8().wrapping_add(self.cpu.x);
        let lo = self.cpu_read(ptr as u16) as u16;
        let hi = self.cpu_read(ptr.wrapping_add(1) as u16) as u16;
        lo | (hi << 8)
    }

    #[inline]
    fn indy(&mut self) -> (u16, bool) {
        let ptr = self.fetch_u8();
        let lo = self.cpu_read(ptr as u16) as u16;
        let hi = self.cpu_read(ptr.wrapping_add(1) as u16) as u16;
        let base = lo | (hi << 8);
        let addr = base.wrapping_add(self.cpu.y as u16);
        (addr, page_crossed(base, addr))
    }

    #[inline]
    fn set_flag(&mut self, flag: u8, value: bool) {
        if value {
            self.cpu.p |= flag;
        } else {
            self.cpu.p &= !flag;
        }
        self.cpu.p |= FLAG_U;
    }

    #[inline]
    fn flag(&self, flag: u8) -> bool {
        self.cpu.p & flag != 0
    }

    #[inline]
    fn set_zn(&mut self, value: u8) {
        let mut p = self.cpu.p & !(FLAG_Z | FLAG_N);
        if value == 0 {
            p |= FLAG_Z;
        }
        if value & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
    }

    #[inline]
    fn push(&mut self, value: u8) {
        let addr = 0x0100 | self.cpu.sp as u16;
        self.cpu_write(addr, value);
        self.cpu.sp = self.cpu.sp.wrapping_sub(1);
    }

    #[inline]
    fn pop(&mut self) -> u8 {
        self.cpu.sp = self.cpu.sp.wrapping_add(1);
        self.cpu_read(0x0100 | self.cpu.sp as u16)
    }

    #[inline]
    fn push_u16(&mut self, value: u16) {
        self.push((value >> 8) as u8);
        self.push(value as u8);
    }

    #[inline]
    fn pop_u16(&mut self) -> u16 {
        let lo = self.pop() as u16;
        let hi = self.pop() as u16;
        lo | (hi << 8)
    }

    fn interrupt(&mut self, vector: u16, brk: bool) {
        self.push_u16(self.cpu.pc);
        let mut p = self.cpu.p | FLAG_U;
        if brk {
            p |= FLAG_B;
        } else {
            p &= !FLAG_B;
        }
        self.push(p);
        self.set_flag(FLAG_I, true);
        self.cpu.pc = self.cpu_read_u16(vector);
    }

    fn cpu_step(&mut self) -> u16 {
        let opcode = self.fetch_u8();
        let mut cycles = match opcode {
            0x00 => {
                self.cpu.pc = self.cpu.pc.wrapping_add(1);
                self.interrupt(0xfffe, true);
                7
            }
            0x01 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.ora(v);
                6
            }
            0x05 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.ora(v);
                3
            }
            0x06 => {
                let a = self.zp();
                self.asl_mem(a);
                5
            }
            0x08 => {
                self.push(self.cpu.p | FLAG_B | FLAG_U);
                3
            }
            0x09 => {
                let v = self.fetch_u8();
                self.ora(v);
                2
            }
            0x0a => {
                self.cpu.a = self.asl(self.cpu.a);
                2
            }
            0x0d => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.ora(v);
                4
            }
            0x0e => {
                let a = self.abs();
                self.asl_mem(a);
                6
            }
            0x10 => self.branch(!self.flag(FLAG_N)),
            0x11 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.ora(v);
                5 + p as u16
            }
            0x15 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.ora(v);
                4
            }
            0x16 => {
                let a = self.zpx();
                self.asl_mem(a);
                6
            }
            0x18 => {
                self.set_flag(FLAG_C, false);
                2
            }
            0x19 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.ora(v);
                4 + p as u16
            }
            0x1d => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.ora(v);
                4 + p as u16
            }
            0x1e => {
                let (a, _) = self.absx();
                self.asl_mem(a);
                7
            }
            0x20 => {
                let a = self.abs();
                self.push_u16(self.cpu.pc.wrapping_sub(1));
                self.cpu.pc = a;
                6
            }
            0x21 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.and(v);
                6
            }
            0x24 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.bit(v);
                3
            }
            0x25 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.and(v);
                3
            }
            0x26 => {
                let a = self.zp();
                self.rol_mem(a);
                5
            }
            0x28 => {
                self.cpu.p = (self.pop() & !FLAG_B) | FLAG_U;
                4
            }
            0x29 => {
                let v = self.fetch_u8();
                self.and(v);
                2
            }
            0x2a => {
                self.cpu.a = self.rol(self.cpu.a);
                2
            }
            0x2c => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.bit(v);
                4
            }
            0x2d => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.and(v);
                4
            }
            0x2e => {
                let a = self.abs();
                self.rol_mem(a);
                6
            }
            0x30 => self.branch(self.flag(FLAG_N)),
            0x31 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.and(v);
                5 + p as u16
            }
            0x35 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.and(v);
                4
            }
            0x36 => {
                let a = self.zpx();
                self.rol_mem(a);
                6
            }
            0x38 => {
                self.set_flag(FLAG_C, true);
                2
            }
            0x39 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.and(v);
                4 + p as u16
            }
            0x3d => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.and(v);
                4 + p as u16
            }
            0x3e => {
                let (a, _) = self.absx();
                self.rol_mem(a);
                7
            }
            0x40 => {
                self.cpu.p = (self.pop() & !FLAG_B) | FLAG_U;
                self.cpu.pc = self.pop_u16();
                6
            }
            0x41 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.eor(v);
                6
            }
            0x45 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.eor(v);
                3
            }
            0x46 => {
                let a = self.zp();
                self.lsr_mem(a);
                5
            }
            0x48 => {
                self.push(self.cpu.a);
                3
            }
            0x49 => {
                let v = self.fetch_u8();
                self.eor(v);
                2
            }
            0x4a => {
                self.cpu.a = self.lsr(self.cpu.a);
                2
            }
            0x4c => {
                let a = self.abs();
                self.cpu.pc = a;
                3
            }
            0x4d => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.eor(v);
                4
            }
            0x4e => {
                let a = self.abs();
                self.lsr_mem(a);
                6
            }
            0x50 => self.branch(!self.flag(FLAG_V)),
            0x51 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.eor(v);
                5 + p as u16
            }
            0x55 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.eor(v);
                4
            }
            0x56 => {
                let a = self.zpx();
                self.lsr_mem(a);
                6
            }
            0x58 => {
                self.set_flag(FLAG_I, false);
                2
            }
            0x59 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.eor(v);
                4 + p as u16
            }
            0x5d => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.eor(v);
                4 + p as u16
            }
            0x5e => {
                let (a, _) = self.absx();
                self.lsr_mem(a);
                7
            }
            0x60 => {
                self.cpu.pc = self.pop_u16().wrapping_add(1);
                6
            }
            0x61 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.adc(v);
                6
            }
            0x65 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.adc(v);
                3
            }
            0x66 => {
                let a = self.zp();
                self.ror_mem(a);
                5
            }
            0x68 => {
                self.cpu.a = self.pop();
                self.set_zn(self.cpu.a);
                4
            }
            0x69 => {
                let v = self.fetch_u8();
                self.adc(v);
                2
            }
            0x6a => {
                self.cpu.a = self.ror(self.cpu.a);
                2
            }
            0x6c => {
                let ptr = self.abs();
                let lo = self.cpu_read(ptr) as u16;
                let hi_addr = (ptr & 0xff00) | ptr.wrapping_add(1) & 0x00ff;
                let hi = self.cpu_read(hi_addr) as u16;
                self.cpu.pc = lo | (hi << 8);
                5
            }
            0x6d => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.adc(v);
                4
            }
            0x6e => {
                let a = self.abs();
                self.ror_mem(a);
                6
            }
            0x70 => self.branch(self.flag(FLAG_V)),
            0x71 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.adc(v);
                5 + p as u16
            }
            0x75 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.adc(v);
                4
            }
            0x76 => {
                let a = self.zpx();
                self.ror_mem(a);
                6
            }
            0x78 => {
                self.set_flag(FLAG_I, true);
                2
            }
            0x79 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.adc(v);
                4 + p as u16
            }
            0x7d => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.adc(v);
                4 + p as u16
            }
            0x7e => {
                let (a, _) = self.absx();
                self.ror_mem(a);
                7
            }
            0x81 => {
                let a = self.indx();
                self.cpu_write(a, self.cpu.a);
                6
            }
            0x84 => {
                let a = self.zp();
                self.cpu_write(a, self.cpu.y);
                3
            }
            0x85 => {
                let a = self.zp();
                self.cpu_write(a, self.cpu.a);
                3
            }
            0x86 => {
                let a = self.zp();
                self.cpu_write(a, self.cpu.x);
                3
            }
            0x88 => {
                self.cpu.y = self.cpu.y.wrapping_sub(1);
                self.set_zn(self.cpu.y);
                2
            }
            0x8a => {
                self.cpu.a = self.cpu.x;
                self.set_zn(self.cpu.a);
                2
            }
            0x8c => {
                let a = self.abs();
                self.cpu_write(a, self.cpu.y);
                4
            }
            0x8d => {
                let a = self.abs();
                self.cpu_write(a, self.cpu.a);
                4
            }
            0x8e => {
                let a = self.abs();
                self.cpu_write(a, self.cpu.x);
                4
            }
            0x90 => self.branch(!self.flag(FLAG_C)),
            0x91 => {
                let (a, _) = self.indy();
                self.cpu_write(a, self.cpu.a);
                6
            }
            0x94 => {
                let a = self.zpx();
                self.cpu_write(a, self.cpu.y);
                4
            }
            0x95 => {
                let a = self.zpx();
                self.cpu_write(a, self.cpu.a);
                4
            }
            0x96 => {
                let a = self.zpy();
                self.cpu_write(a, self.cpu.x);
                4
            }
            0x98 => {
                self.cpu.a = self.cpu.y;
                self.set_zn(self.cpu.a);
                2
            }
            0x99 => {
                let (a, _) = self.absy();
                self.cpu_write(a, self.cpu.a);
                5
            }
            0x9a => {
                self.cpu.sp = self.cpu.x;
                2
            }
            0x9d => {
                let (a, _) = self.absx();
                self.cpu_write(a, self.cpu.a);
                5
            }
            0xa0 => {
                let v = self.fetch_u8();
                self.cpu.y = v;
                self.set_zn(v);
                2
            }
            0xa1 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                6
            }
            0xa2 => {
                let v = self.fetch_u8();
                self.cpu.x = v;
                self.set_zn(v);
                2
            }
            0xa4 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.cpu.y = v;
                self.set_zn(v);
                3
            }
            0xa5 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                3
            }
            0xa6 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.cpu.x = v;
                self.set_zn(v);
                3
            }
            0xa8 => {
                self.cpu.y = self.cpu.a;
                self.set_zn(self.cpu.y);
                2
            }
            0xa9 => {
                let v = self.fetch_u8();
                self.cpu.a = v;
                self.set_zn(v);
                2
            }
            0xaa => {
                self.cpu.x = self.cpu.a;
                self.set_zn(self.cpu.x);
                2
            }
            0xac => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cpu.y = v;
                self.set_zn(v);
                4
            }
            0xad => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                4
            }
            0xae => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cpu.x = v;
                self.set_zn(v);
                4
            }
            0xb0 => self.branch(self.flag(FLAG_C)),
            0xb1 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                5 + p as u16
            }
            0xb4 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.cpu.y = v;
                self.set_zn(v);
                4
            }
            0xb5 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                4
            }
            0xb6 => {
                let a = self.zpy();
                let v = self.cpu_read(a);
                self.cpu.x = v;
                self.set_zn(v);
                4
            }
            0xb8 => {
                self.set_flag(FLAG_V, false);
                2
            }
            0xb9 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                4 + p as u16
            }
            0xba => {
                self.cpu.x = self.cpu.sp;
                self.set_zn(self.cpu.x);
                2
            }
            0xbc => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.cpu.y = v;
                self.set_zn(v);
                4 + p as u16
            }
            0xbd => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                4 + p as u16
            }
            0xbe => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.cpu.x = v;
                self.set_zn(v);
                4 + p as u16
            }
            0xc0 => {
                let v = self.fetch_u8();
                self.cmp(self.cpu.y, v);
                2
            }
            0xc1 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                6
            }
            0xc4 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.y, v);
                3
            }
            0xc5 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                3
            }
            0xc6 => {
                let a = self.zp();
                self.dec_mem(a);
                5
            }
            0xc8 => {
                self.cpu.y = self.cpu.y.wrapping_add(1);
                self.set_zn(self.cpu.y);
                2
            }
            0xc9 => {
                let v = self.fetch_u8();
                self.cmp(self.cpu.a, v);
                2
            }
            0xca => {
                self.cpu.x = self.cpu.x.wrapping_sub(1);
                self.set_zn(self.cpu.x);
                2
            }
            0xcc => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.y, v);
                4
            }
            0xcd => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                4
            }
            0xce => {
                let a = self.abs();
                self.dec_mem(a);
                6
            }
            0xd0 => self.branch(!self.flag(FLAG_Z)),
            0xd1 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                5 + p as u16
            }
            0xd5 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                4
            }
            0xd6 => {
                let a = self.zpx();
                self.dec_mem(a);
                6
            }
            0xd8 => {
                self.set_flag(FLAG_D, false);
                2
            }
            0xd9 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                4 + p as u16
            }
            0xdd => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                4 + p as u16
            }
            0xde => {
                let (a, _) = self.absx();
                self.dec_mem(a);
                7
            }
            0xe0 => {
                let v = self.fetch_u8();
                self.cmp(self.cpu.x, v);
                2
            }
            0xe1 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.sbc(v);
                6
            }
            0xe4 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.x, v);
                3
            }
            0xe5 => {
                let a = self.zp();
                let v = self.cpu_read(a);
                self.sbc(v);
                3
            }
            0xe6 => {
                let a = self.zp();
                self.inc_mem(a);
                5
            }
            0xe8 => {
                self.cpu.x = self.cpu.x.wrapping_add(1);
                self.set_zn(self.cpu.x);
                2
            }
            0xe9 => {
                let v = self.fetch_u8();
                self.sbc(v);
                2
            }
            0xea => 2,
            0xec => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.x, v);
                4
            }
            0xed => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.sbc(v);
                4
            }
            0xee => {
                let a = self.abs();
                self.inc_mem(a);
                6
            }
            0xf0 => self.branch(self.flag(FLAG_Z)),
            0xf1 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.sbc(v);
                5 + p as u16
            }
            0xf5 => {
                let a = self.zpx();
                let v = self.cpu_read(a);
                self.sbc(v);
                4
            }
            0xf6 => {
                let a = self.zpx();
                self.inc_mem(a);
                6
            }
            0xf8 => {
                self.set_flag(FLAG_D, true);
                2
            }
            0xf9 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.sbc(v);
                4 + p as u16
            }
            0xfd => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.sbc(v);
                4 + p as u16
            }
            0xfe => {
                let (a, _) = self.absx();
                self.inc_mem(a);
                7
            }
            _ => 2,
        };
        cycles = cycles.saturating_add(self.extra_cycles);
        self.extra_cycles = 0;
        cycles
    }

    #[inline]
    fn branch(&mut self, condition: bool) -> u16 {
        let offset = self.fetch_u8() as i8;
        if !condition {
            return 2;
        }
        let old_pc = self.cpu.pc;
        self.cpu.pc = self.cpu.pc.wrapping_add(offset as i16 as u16);
        3 + page_crossed(old_pc, self.cpu.pc) as u16
    }

    #[inline]
    fn adc(&mut self, value: u8) {
        let carry = u8::from(self.flag(FLAG_C));
        let a = self.cpu.a;
        let sum = a as u16 + value as u16 + carry as u16;
        let result = sum as u8;
        self.cpu.a = result;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_V | FLAG_Z | FLAG_N);
        if sum > 0xff {
            p |= FLAG_C;
        }
        if (!(a ^ value) & (a ^ result) & 0x80) != 0 {
            p |= FLAG_V;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
    }

    #[inline]
    fn sbc(&mut self, value: u8) {
        self.adc(!value);
    }

    #[inline]
    fn cmp(&mut self, reg: u8, value: u8) {
        let result = reg.wrapping_sub(value);
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if reg >= value {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
    }

    #[inline]
    fn ora(&mut self, value: u8) {
        self.cpu.a |= value;
        self.set_zn(self.cpu.a);
    }

    #[inline]
    fn and(&mut self, value: u8) {
        self.cpu.a &= value;
        self.set_zn(self.cpu.a);
    }

    #[inline]
    fn eor(&mut self, value: u8) {
        self.cpu.a ^= value;
        self.set_zn(self.cpu.a);
    }

    #[inline]
    fn bit(&mut self, value: u8) {
        let mut p = self.cpu.p & !(FLAG_Z | FLAG_V | FLAG_N);
        if self.cpu.a & value == 0 {
            p |= FLAG_Z;
        }
        p |= value & (FLAG_V | FLAG_N);
        self.cpu.p = p | FLAG_U;
    }

    #[inline]
    fn asl(&mut self, value: u8) -> u8 {
        let result = value << 1;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if value & 0x80 != 0 {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
        result
    }

    #[inline]
    fn lsr(&mut self, value: u8) -> u8 {
        let result = value >> 1;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if value & 1 != 0 {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        self.cpu.p = p | FLAG_U;
        result
    }

    #[inline]
    fn rol(&mut self, value: u8) -> u8 {
        let carry = u8::from(self.flag(FLAG_C));
        let result = (value << 1) | carry;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if value & 0x80 != 0 {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
        result
    }

    #[inline]
    fn ror(&mut self, value: u8) -> u8 {
        let carry = if self.flag(FLAG_C) { 0x80 } else { 0 };
        let result = (value >> 1) | carry;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if value & 1 != 0 {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
        result
    }

    #[inline]
    fn asl_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.asl(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    fn lsr_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.lsr(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    fn rol_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.rol(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    fn ror_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.ror(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    fn dec_mem(&mut self, addr: u16) {
        let result = self.cpu_read(addr).wrapping_sub(1);
        self.cpu_write(addr, result);
        self.set_zn(result);
    }

    #[inline]
    fn inc_mem(&mut self, addr: u16) {
        let result = self.cpu_read(addr).wrapping_add(1);
        self.cpu_write(addr, result);
        self.set_zn(result);
    }
}

#[inline]
fn page_crossed(a: u16, b: u16) -> bool {
    (a & 0xff00) != (b & 0xff00)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resize_default_area_reference(src: &[u8], dst: &mut [u8]) {
        for dst_y in 0..DEFAULT_GRAY_RESIZE_HEIGHT {
            let y0 = (dst_y * DEFAULT_GRAY_CROP_HEIGHT) / DEFAULT_GRAY_RESIZE_HEIGHT;
            let y1 = (((dst_y + 1) * DEFAULT_GRAY_CROP_HEIGHT) / DEFAULT_GRAY_RESIZE_HEIGHT)
                .max(y0 + 1)
                .min(DEFAULT_GRAY_CROP_HEIGHT);
            for dst_x in 0..DEFAULT_GRAY_RESIZE_WIDTH {
                let x0 = (dst_x * VISIBLE_FRAME_WIDTH) / DEFAULT_GRAY_RESIZE_WIDTH;
                let x1 = (((dst_x + 1) * VISIBLE_FRAME_WIDTH) / DEFAULT_GRAY_RESIZE_WIDTH)
                    .max(x0 + 1)
                    .min(VISIBLE_FRAME_WIDTH);
                let mut sum = 0u32;
                for sy in y0..y1 {
                    let row = sy * VISIBLE_FRAME_WIDTH;
                    for sx in x0..x1 {
                        sum += src[row + sx] as u32;
                    }
                }
                dst[dst_y * DEFAULT_GRAY_RESIZE_WIDTH + dst_x] =
                    (sum / ((x1 - x0) * (y1 - y0)) as u32) as u8;
            }
        }
    }

    fn set_sprite(ppu: &mut Ppu, sprite: usize, y: u8, tile: u8, attr: u8, x: u8) {
        let base = sprite * 4;
        ppu.oam[base] = y.wrapping_sub(1);
        ppu.oam[base + 1] = tile;
        ppu.oam[base + 2] = attr;
        ppu.oam[base + 3] = x;
    }

    fn make_test_ppu() -> Ppu {
        let chr_rom = (0..8192)
            .map(|idx| ((idx * 37 + idx / 11 + 23) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut ppu = Ppu::new(chr_rom, true);
        ppu.ctrl = 0x18;
        ppu.mask = 0x18;
        ppu.scroll_x_px = 37;
        ppu.scroll_x_low = 37;
        ppu.scroll_y_px = 11;
        for (idx, value) in ppu.vram.iter_mut().enumerate() {
            *value = ((idx * 13 + idx / 7 + 5) & 0xff) as u8;
        }
        for (idx, value) in ppu.palette.iter_mut().enumerate() {
            *value = ((idx * 3 + 7) & 0x3f) as u8;
        }
        ppu.oam.fill(0xff);
        set_sprite(&mut ppu, 63, 70, 3, 0x00, 40);
        set_sprite(&mut ppu, 0, 72, 5, 0x01, 42);
        set_sprite(&mut ppu, 1, 74, 7, 0x22, 44);
        set_sprite(&mut ppu, 2, 190, 9, 0xc3, 250);
        ppu
    }

    #[test]
    fn sprite_scanline_mask_limits_to_first_eight_oam_sprites() {
        let mut ppu = Ppu::new(vec![0; 8192], true);
        ppu.oam.fill(0xff);
        for sprite in 0..9usize {
            set_sprite(&mut ppu, sprite, 50, 1, 0x00, (sprite * 8) as u8);
        }

        let mask = ppu.sprite_scanline_mask();

        assert_eq!(mask[50].count_ones(), 8);
        for sprite in 0..8usize {
            assert_ne!(mask[50] & (1u64 << sprite), 0);
        }
        assert_eq!(mask[50] & (1u64 << 8), 0);
    }

    #[test]
    fn behind_background_sprite_blocks_lower_priority_sprites() {
        let mut chr_rom = vec![0; 8192];
        for tile in [1usize, 2usize] {
            for row in 0..8usize {
                chr_rom[tile * 16 + row] = 0xff;
            }
        }
        let mut ppu = Ppu::new(chr_rom, true);
        ppu.mask = 0x18;
        ppu.vram[(40 / 8) * 32 + (40 / 8)] = 2;
        ppu.palette[1] = 0x0f;
        ppu.palette[0x11] = 0x30;
        ppu.oam.fill(0xff);
        set_sprite(&mut ppu, 1, 40, 1, 0x00, 40);
        set_sprite(&mut ppu, 0, 40, 1, 0x20, 40);

        let mut dst =
            vec![NES_GRAY_PALETTE[ppu.bg_color_index(40, 40) as usize]; NES_WIDTH * NES_HEIGHT];
        ppu.draw_sprites_gray(&mut dst);

        assert_eq!(
            dst[40 * NES_WIDTH + 40],
            NES_GRAY_PALETTE[ppu.bg_color_index(40, 40) as usize],
        );
    }

    fn assert_default_area_writer_matches_scratch(ppu: &Ppu) {
        let mut native = vec![0; VISIBLE_FRAME_WIDTH * DEFAULT_GRAY_CROP_HEIGHT];
        let mut expected = vec![0; DEFAULT_GRAY_RESIZE_PIXELS];
        let mut actual = vec![0; DEFAULT_GRAY_RESIZE_PIXELS];
        let mut sprite_shadow = vec![0; VISIBLE_FRAME_WIDTH * DEFAULT_GRAY_CROP_HEIGHT];

        ppu.write_gray_frame_region(
            &mut native,
            VISIBLE_FRAME_TOP + DEFAULT_GRAY_CROP_TOP,
            VISIBLE_FRAME_LEFT,
            VISIBLE_FRAME_WIDTH,
            DEFAULT_GRAY_CROP_HEIGHT,
        );
        resize_default_area_reference(&native, &mut expected);
        ppu.write_gray_frame_cropped_area_84x84(&mut actual, &mut sprite_shadow);

        assert_eq!(actual, expected);
    }

    #[test]
    fn direct_nametable_read_matches_ppu_read_mirroring() {
        for vertical_mirroring in [true, false] {
            let chr_rom = vec![0; 8192];
            let mut ppu = Ppu::new(chr_rom, vertical_mirroring);
            for (idx, value) in ppu.vram.iter_mut().enumerate() {
                *value = ((idx * 29 + idx / 5 + 17) & 0xff) as u8;
            }

            for table in 0..4usize {
                for offset in 0..0x400usize {
                    let addr = 0x2000 + (table * 0x400 + offset) as u16;
                    assert_eq!(ppu.nametable_read(table, offset), ppu.ppu_read(addr));
                }
            }
        }
    }

    #[test]
    fn default_cropped_gray_area_writer_matches_scratch_resize() {
        let mut ppu = make_test_ppu();
        assert_default_area_writer_matches_scratch(&ppu);

        ppu.mask = 0x10;
        assert_default_area_writer_matches_scratch(&ppu);

        ppu.mask = 0x00;
        assert_default_area_writer_matches_scratch(&ppu);
    }
}
