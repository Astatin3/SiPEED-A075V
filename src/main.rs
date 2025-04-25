use byteorder::{LittleEndian, ReadBytesExt};
use opencv::boxed_ref::BoxedRef;
use opencv::{core, highgui, imgcodecs, imgproc, prelude::*};
use reqwest;
use std::error::Error;
use std::io::Cursor;

const HOST: &str = "192.168.233.1"; // Assuming default value similar to Python code
const PORT: u16 = 80; // Assuming default value similar to Python code

fn get_frame_from_http() -> Result<Vec<u8>, Box<dyn Error>> {
    let client = reqwest::blocking::Client::new();
    let url = format!("http://{}:{}/getdeep", HOST, PORT);
    let response = client.get(&url).send()?;

    if response.status().is_success() {
        println!("Get deep image");
        let deepimg = response.bytes()?.to_vec();
        println!("Length={}", deepimg.len());

        if deepimg.len() >= 16 {
            let mut cursor = Cursor::new(&deepimg[0..16]);
            let frameid = cursor.read_u64::<LittleEndian>()?;
            let stamp_msec = cursor.read_u64::<LittleEndian>()?;
            println!("({}, {})", frameid, stamp_msec as f64 / 1000.0);

            Ok(deepimg)
        } else {
            Err("Image data too short".into())
        }
    } else {
        Err(format!("HTTP request failed with status: {}", response.status()).into())
    }
}

fn post_encode_config(config: &[u8]) -> Result<bool, Box<dyn Error>> {
    let client = reqwest::blocking::Client::new();
    let url = format!("http://{}:{}/set_cfg", HOST, PORT);
    let response = client.post(&url).body(config.to_vec()).send()?;

    Ok(response.status().is_success())
}

#[derive(Debug, Clone, Copy)]
struct FrameConfig {
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

fn frame_config_decode(frame_config: &[u8]) -> Result<FrameConfig, Box<dyn Error>> {
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

fn frame_config_encode(
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
    let mut buffer = Vec::with_capacity(12);
    buffer.push(trigger_mode);
    buffer.push(deep_mode);
    buffer.push(deep_shift);
    buffer.push(ir_mode);
    buffer.push(status_mode);
    buffer.push(status_mask);
    buffer.push(rgb_mode);
    buffer.push(rgb_res);

    buffer.extend_from_slice(&expose_time.to_le_bytes());

    buffer
}

struct FramePayload {
    depth_img: Option<Vec<u8>>,
    ir_img: Option<Vec<u8>>,
    status_img: Option<Vec<u8>>,
    rgb_img: Option<Vec<u8>>,
}

fn frame_payload_decode(
    frame_data: &[u8],
    config: &FrameConfig,
) -> Result<FramePayload, Box<dyn Error>> {
    if frame_data.len() < 8 {
        return Err("Frame data too short".into());
    }

    let mut cursor = Cursor::new(&frame_data[0..8]);
    let deep_data_size = cursor.read_i32::<LittleEndian>()?;
    let rgb_data_size = cursor.read_i32::<LittleEndian>()?;

    let mut frame_payload = &frame_data[8..];

    // Calculate sizes based on configuration
    let depth_size = (320 * 240 * 2) >> config.deep_mode;
    let depth_img = if depth_size > 0 && frame_payload.len() >= depth_size {
        let depth_data = frame_payload[..depth_size].to_vec();
        frame_payload = &frame_payload[depth_size..];
        Some(depth_data)
    } else {
        None
    };

    let ir_size = (320 * 240 * 2) >> config.ir_mode;
    let ir_img = if ir_size > 0 && frame_payload.len() >= ir_size {
        let ir_data = frame_payload[..ir_size].to_vec();
        frame_payload = &frame_payload[ir_size..];
        Some(ir_data)
    } else {
        None
    };

    let status_size = (320 * 240 / 8)
        * match config.status_mode {
            0 => 16,
            1 => 2,
            2 => 8,
            _ => 1,
        };

    let status_img = if status_size > 0 && frame_payload.len() >= status_size {
        let status_data = frame_payload[..status_size].to_vec();
        frame_payload = &frame_payload[status_size..];
        Some(status_data)
    } else {
        None
    };

    // Verify that we've consumed the expected amount of data
    if deep_data_size as usize != depth_size + ir_size + status_size {
        return Err("Data size mismatch in frame payload".into());
    }

    let rgb_size = frame_payload.len();
    if rgb_data_size as usize != rgb_size {
        return Err("RGB data size mismatch".into());
    }

    let mut rgb_img = if rgb_size > 0 {
        Some(frame_payload.to_vec())
    } else {
        None
    };

    // Process RGB image if present
    if let Some(rgb_data) = rgb_img.as_ref() {
        if config.rgb_mode == 1 {
            let jpeg_data = Mat::from_slice(rgb_data)?;
            let jpeg = imgcodecs::imdecode(&jpeg_data, imgcodecs::IMREAD_COLOR)?;

            if !jpeg.empty() {
                let mut rgb = Mat::default();
                imgproc::cvt_color(&jpeg, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

                let mut rgb_vec = Vec::new();
                let mat_size = rgb.total() as usize * rgb.elem_size().unwrap();
                rgb_vec.resize(mat_size, 0);

                // This extracts the raw bytes
                let mat_data = rgb.data_bytes()?;
                rgb_vec.copy_from_slice(mat_data);

                rgb_img = Some(rgb_vec);
            } else {
                rgb_img = None;
            }
        }
    }

    Ok(FramePayload {
        depth_img,
        ir_img,
        status_img,
        rgb_img,
    })
}

struct FrameOutputs {
    depth: Option<BoxedRef<'static, Mat>>,
    ir: Option<BoxedRef<'static, Mat>>,
    status: Option<BoxedRef<'static, Mat>>,
    rgb: Option<BoxedRef<'static, Mat>>,
}

fn show_frame(frame_data: &[u8]) -> Result<(), Box<dyn Error>> {
    if frame_data.len() < 28 {
        // 16 + 12 minimum size
        return Err("Frame data too short".into());
    }

    let config = frame_config_decode(&frame_data[16..28])?;
    let frame_bytes = frame_payload_decode(&frame_data[28..], &config)?;

    let depth = if let Some(depth_data) = frame_bytes.depth_img {
        let depth_type = if config.deep_mode == 0 {
            core::CV_16U
        } else {
            core::CV_8U
        };

        // Create a Mat from raw depth data
        let depth_mat = unsafe {
            let mut mat = Mat::new_rows_cols(240, 320, depth_type)?;
            let mat_data = mat.data_bytes_mut()?;
            if mat_data.len() == depth_data.len() {
                mat_data.copy_from_slice(&depth_data);
                mat
            } else {
                return Err("Depth data size mismatch".into());
            }
        };

        // highgui::imshow("depth_mat", &depth_mat)?;

        Some(depth_mat)
    } else {
        None
    };

    let ir = if let Some(ir_data) = frame_bytes.ir_img {
        let ir_type = if config.ir_mode == 0 {
            core::CV_16U
        } else {
            core::CV_8U
        };

        // Create a Mat from raw IR data
        let mut ir_mat = unsafe {
            let mut mat = Mat::new_rows_cols(240, 320, ir_type)?;
            let mat_data = mat.data_bytes_mut()?;
            if mat_data.len() == ir_data.len() {
                mat_data.copy_from_slice(&ir_data);
                mat
            } else {
                return Err("IR data size mismatch".into());
            }
        };

        // highgui::imshow("ir_mat", &ir_mat)?;

        Some(ir_mat)
    } else {
        None
    };

    // Process status image
    let status = if let Some(status_data) = frame_bytes.status_img {
        let status_type = if config.status_mode == 0 {
            core::CV_16U
        } else {
            core::CV_8U
        };

        // Create a Mat from raw status data
        let mut status_mat = unsafe {
            let mut mat = Mat::new_rows_cols(240, 320, status_type)?;
            let mat_data = mat.data_bytes_mut()?;
            if mat_data.len() == status_data.len() {
                mat_data.copy_from_slice(&status_data);
                mat
            } else {
                return Err("Status data size mismatch".into());
            }
        };

        // highgui::imshow("status_mat", &status_mat)?;

        Some(status_mat)
    } else {
        None
    };

    // Process IR image
    // let ir = if let Some(ir_data) = &frame_bytes.ir_img {
    //     let ir_data = ir_data.as_slice();
    //     let ir_mat = Mat::new_rows_cols_with_data(240, 320, ir_data).unwrap();

    //     Some(ir_mat)
    // } else {
    //     None
    // };

    // Process RGB image
    let rgb = if let Some(rgb_data) = &frame_bytes.rgb_img {
        let (height, width) = if config.rgb_mode == 1 {
            (480, 640)
        } else {
            (600, 800)
        };

        let rgb_mat = unsafe {
            let mut mat = Mat::new_rows_cols(height, width, core::CV_8UC3)?;
            let mat_data = mat.data_bytes_mut()?;
            if mat_data.len() == rgb_data.len() {
                mat_data.copy_from_slice(&rgb_data);
                mat
            } else {
                return Err("RGB data size mismatch".into());
            }
        };

        highgui::imshow("rgb", &rgb_mat);

        Some(rgb_mat)
    } else {
        None
    };

    Ok(())

    // Ok(FrameOutputs {
    //     depth,
    //     ir,
    //     status,
    //     rgb,
    // })
}

fn scanloop() -> Result<(), Box<dyn Error>> {
    if post_encode_config(&frame_config_encode(1, 0, 255, 0, 2, 7, 1, 0, 0)).unwrap_or(false) {
        let raw_frame = get_frame_from_http()?;
        show_frame(&raw_frame)?;
        highgui::wait_key(1)?;
    }

    Ok(())
}

fn main() {
    loop {
        let _ = scanloop();
    }
}
