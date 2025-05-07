use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::{Array2, Array3};
use std::{io::Cursor, ops::DerefMut};

pub struct ProcessedFrames {
    pub depth: Option<Array2<u16>>,
    pub ir: Option<Array2<u16>>,
    pub status: Option<Array2<u16>>,
    pub rgb: Option<Array3<u8>>,
}

impl Default for ProcessedFrames {
    fn default() -> Self {
        Self {
            depth: None,
            ir: None,
            status: None,
            rgb: None,
        }
    }
}

// Messages between threads
pub enum FrameMessage {
    RawFrame(Vec<u8>),
    DecodedFrame(ProcessedFrames),
    Shutdown,
}

// Frame data structures
#[allow(dead_code)]
pub struct FrameConfig {
    trigger_mode: u8,
    deep_mode: u8,
    deep_shift: u8,
    ir_mode: u8,
    status_mode: u8,
    status_mask: u8,
    rgb_mode: u8,
    rgb_res: u8,
    expose_time: i32,
}

pub struct FramePayload {
    depth_img: Option<Vec<u8>>,
    ir_img: Option<Vec<u8>>,
    status_img: Option<Vec<u8>>,
    rgb_img: Option<Vec<u8>>,
}

// Helper functions
fn frame_config_decode(frame_config: &[u8]) -> Result<FrameConfig, Box<dyn std::error::Error>> {
    if frame_config.len() < 12 {
        return Err("Frame config data too short".into());
    }

    let mut cursor = Cursor::new(frame_config);

    Ok(FrameConfig {
        trigger_mode: cursor.read_u8()?,
        deep_mode: cursor.read_u8()?,
        deep_shift: cursor.read_u8()?,
        ir_mode: cursor.read_u8()?,
        status_mode: cursor.read_u8()?,
        status_mask: cursor.read_u8()?,
        rgb_mode: cursor.read_u8()?,
        rgb_res: cursor.read_u8()?,
        expose_time: cursor.read_i32::<LittleEndian>()?,
    })
}

pub fn frame_config_encode(
    trigger_mode: u8,
    deep_mode: u8,
    deep_shift: u8,
    ir_mode: u8,
    status_mode: u8,
    status_mask: u8,
    rgb_mode: u8,
    rgb_res: u8,
    expose_time: i32,
) -> Vec<u8> {
    let mut result = Vec::with_capacity(12);
    result.push(trigger_mode);
    result.push(deep_mode);
    result.push(deep_shift);
    result.push(ir_mode);
    result.push(status_mode);
    result.push(status_mask);
    result.push(rgb_mode);
    result.push(rgb_res);

    // Add expose_time as little endian
    result.extend_from_slice(&expose_time.to_le_bytes());

    result
}

fn frame_payload_decode(
    frame_data: &[u8],
    config: &FrameConfig,
) -> Result<FramePayload, Box<dyn std::error::Error>> {
    if frame_data.len() < 8 {
        return Err("Frame data too short".into());
    }

    let mut cursor = Cursor::new(&frame_data[0..8]);
    let deep_data_size = cursor.read_i32::<LittleEndian>()?;
    let rgb_data_size = cursor.read_i32::<LittleEndian>()?;

    let mut payload = &frame_data[8..];

    // Depth image
    let depth_size = (320 * 240 * 2) >> config.deep_mode;
    let depth_img = if depth_size > 0 && payload.len() >= depth_size {
        let result = payload[..depth_size].to_vec();
        payload = &payload[depth_size..];
        Some(result)
    } else {
        None
    };

    // IR image
    let ir_size = (320 * 240 * 2) >> config.ir_mode;
    let ir_img = if ir_size > 0 && payload.len() >= ir_size {
        let result = payload[..ir_size].to_vec();
        payload = &payload[ir_size..];
        Some(result)
    } else {
        None
    };

    // Status image
    let status_size = (320 * 240 / 8)
        * match config.status_mode {
            0 => 16,
            1 => 2,
            2 => 8,
            _ => 1,
        };

    let status_img = if status_size > 0 && payload.len() >= status_size {
        let result = payload[..status_size].to_vec();
        payload = &payload[status_size..];
        Some(result)
    } else {
        None
    };

    // Verify deep data size
    let calculated_deep_size = depth_size + ir_size + status_size;
    if calculated_deep_size != deep_data_size as usize {
        warn!(
            "Warning: Deep data size mismatch: {} vs {}",
            calculated_deep_size, deep_data_size
        );
    }

    // RGB image
    let rgb_size = payload.len();
    if rgb_size != rgb_data_size as usize {
        warn!(
            "Warning: RGB data size mismatch: {} vs {}",
            rgb_size, rgb_data_size
        );
    }

    let rgb_img = if rgb_size > 0 {
        // Process RGB image based on config
        if config.rgb_mode == 1 {
            // JPEG decode using OpenCV
            let rgb_data = payload.to_vec();
            let decoded = decode_jpeg(&rgb_data);
            decoded
        } else {
            Some(payload.to_vec())
        }
    } else {
        None
    };

    Ok(FramePayload {
        depth_img,
        ir_img,
        status_img,
        rgb_img,
    })
}

fn decode_jpeg(jpeg_data: &[u8]) -> Option<Vec<u8>> {
    // Use the image crate to decode JPEG and convert to RGB
    let img = image::load_from_memory(jpeg_data).ok()?;
    let rgb_img = img.to_rgb8();

    // Convert to raw bytes
    Some(rgb_img.into_raw())
}

pub fn decode_frame(frame_data: &[u8]) -> Result<ProcessedFrames, Box<dyn std::error::Error>> {
    if frame_data.len() < 28 {
        // 16 (header) + 12 (config)
        return Err("Frame data too short".into());
    }

    // Extract config
    let config = frame_config_decode(&frame_data[16..28])?;

    // Decode payload
    let payload = frame_payload_decode(&frame_data[28..], &config)?;

    // Process depth image
    let depth = if let Some(depth_data) = payload.depth_img {
        if config.deep_mode == 0 {
            let data = depth_data.as_slice();
            let depth_array = Array2::from_shape_fn((240, 320), |(y, x)| {
                let idx = (y * 320 + x) * 2;
                if idx + 1 < data.len() {
                    u16::from_le_bytes([data[idx], data[idx + 1]])
                } else {
                    0
                }
            });
            Some(depth_array)
        } else {
            let data = depth_data.as_slice();
            let depth_array = Array2::from_shape_fn((240, 320), |(y, x)| {
                let idx = y * 320 + x;
                if idx < data.len() {
                    u16::from(data[idx])
                } else {
                    0
                }
            });
            Some(depth_array)
        }
    } else {
        None
    };

    // Process IR image
    let ir = if let Some(ir_data) = payload.ir_img {
        if config.ir_mode == 0 {
            let data = ir_data.as_slice();
            let ir_array = Array2::from_shape_fn((240, 320), |(y, x)| {
                let idx = (y * 320 + x) * 2;
                if idx + 1 < data.len() {
                    u16::from_le_bytes([data[idx], data[idx + 1]])
                } else {
                    0
                }
            });
            Some(ir_array)
        } else {
            let data = ir_data.as_slice();
            let ir_array = Array2::from_shape_fn((240, 320), |(y, x)| {
                let idx = y * 320 + x;
                if idx < data.len() {
                    u16::from(data[idx])
                } else {
                    0
                }
            });
            Some(ir_array)
        }
    } else {
        None
    };

    // Process status image
    let status = if let Some(status_data) = payload.status_img {
        // Process according to status_mode
        let data = status_data.as_slice();
        let status_array = Array2::from_shape_fn((240, 320), |(y, x)| {
            // This is a simplified approach - actual processing depends on status_mode
            let idx = y * 320 + x;
            if idx < data.len() {
                u16::from(data[idx])
            } else {
                0
            }
        });
        Some(status_array)
    } else {
        None
    };

    // Process RGB image
    let rgb = if let Some(rgb_data) = payload.rgb_img {
        let shape = if config.rgb_mode == 1 {
            match config.rgb_res {
                0 => (480, 640, 3), // Default resolution
                _ => (600, 800, 3), // Alternative resolution
            }
        } else {
            (480, 640, 3) // Default for non-JPEG
        };

        if rgb_data.len() >= shape.0 * shape.1 * shape.2 {
            let rgb_array = Array3::from_shape_vec((shape.0, shape.1, shape.2), rgb_data)?;
            Some(rgb_array)
        } else {
            warn!(
                "RGB data size ({}) doesn't match expected size ({})",
                rgb_data.len(),
                shape.0 * shape.1 * shape.2
            );
            None
        }
    } else {
        None
    };

    Ok(ProcessedFrames {
        depth,
        ir,
        status,
        rgb,
    })
}

pub fn normalize(data: &Array2<u16>) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.dim().0 * data.dim().1 * 3);

    let max = 255. / (*data.iter().max().unwrap_or(&255u16) as f32);

    for &value in data.iter() {
        let num = (value as f32 * max) as u8;

        result.push(num);
        result.push(num);
        result.push(num);
    }

    result
}
