use std::{
    mem,
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender},
    },
    thread,
};

use std::io::Cursor;

use byteorder::{LittleEndian, ReadBytesExt};
use egui::{Color32, Ui, mutex::Mutex};

use crate::{
    fetch_frame::{FrameMessage, ProcessedFrames, decode_frame, frame_config_encode, normalize},
    pane_manager::{Pane, PaneMode, PaneState, PsudoCreationContext},
    panes::point_cloud_renderer::LivePointView,
};

// Constants (replace with your actual values)
const HOST: &str = "192.168.233.1";
const PORT: u16 = 80;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct CameraPane {
    #[serde(skip)]
    frames: Arc<Mutex<ProcessedFrames>>,
    #[serde(skip)]
    thread_handle: Option<thread::JoinHandle<()>>,
    #[serde(skip)]
    point_cloud: LivePointView,
}

impl Default for CameraPane {
    fn default() -> Self {
        // Shared state for the latest processed frames
        let frames = Arc::new(Mutex::new(ProcessedFrames::default()));

        let frames_clone = Arc::clone(&frames);
        let decoder_handle = thread::spawn(move || {
            loop {
                match fetch_frame() {
                    Ok(frame_data) => match decode_frame(&frame_data) {
                        Ok(processed) => {
                            mem::replace(&mut *frames_clone.lock(), processed);
                        }
                        Err(e) => warn!("Error decoding frame: {}", e),
                    },
                    Err(e) => warn!("Error fetching frame: {}", e),
                }
            }
        });

        Self {
            frames,
            thread_handle: Some(decoder_handle),
            point_cloud: LivePointView::default(),
        }
    }
}

#[typetag::serde]
impl Pane for CameraPane {
    fn new() -> PaneState {
        let mut s = CameraPane::default();
        PaneState {
            id: s.name().to_string(),
            mode: PaneMode::Hidden,
            pane: Box::new(s),
        }
    }

    fn init(&mut self, pcc: &PsudoCreationContext) {
        self.point_cloud.init(pcc);
        // decoder_thread(rx, tx);

        // let mut frames_lock = self.frames.lock();
    }

    fn name(&mut self) -> &str {
        "Raw Camera"
    }

    fn render(&mut self, ui: &mut Ui) {
        let frames_lock = self.frames.lock();
        let ref frames = *frames_lock;

        let mut processed = false;

        // Display depth image
        if let Some(ref depth) = frames.depth {
            let depth_viz = normalize(depth);
            ui.heading("Depth");
            image_widget(ui, "depth_img", &depth_viz, [320.0, 240.0]);
            processed = true;
        }

        // Display IR image
        if let Some(ref ir) = frames.ir {
            let ir_viz = normalize(ir);
            ui.heading("IR");
            image_widget(ui, "ir_img", &ir_viz, [320.0, 240.0]);
            processed = true;
        }

        // Display status image if available
        if let Some(ref status) = frames.status {
            let status_viz = normalize(status);
            ui.heading("Status");
            image_widget(ui, "status_img", &status_viz, [320.0, 240.0]);
            processed = true;
        }

        // Display RGB image if available
        if let Some(ref rgb) = frames.rgb {
            let rgb_viz = rgb.as_slice().unwrap();
            let size = match rgb.dim().1 {
                640 => [640.0, 480.0],
                800 => [800.0, 600.0],
                _ => [640.0, 480.0], // Default
            };
            ui.heading("RGB");
            image_widget(ui, "rgb_img", rgb_viz, size);
            processed = true;
        }

        if !processed {
            ui.heading("Waiting for frames...");
        }

        if let Some(ref status) = frames.status {
            if let Some(ref depth) = frames.depth {
                if let Some(ref rgb) = frames.rgb {
                    let mut points: Vec<(i32, i32, i32, Color32)> = Vec::new();

                    for i in 0..(320 * 240) {
                        let x = i / 240;
                        let y = i % 320;

                        // println!("{:?}", status.get((x, y)));

                        let status = status.get((x, y));

                        if status.is_none() {
                            continue;
                        }

                        let status = *status.unwrap();

                        if status != 0 {
                            continue;
                        }

                        // if *status.get((x, y)).unwrap() != 0u16 {
                        //     continue;
                        // }
                        //

                        let (r, g, b) = if let Some((rgbx, rgby)) = scale_shift_rgb_xy(x, y) {
                            (
                                *rgb.get((rgbx, rgby, 0)).unwrap() as u8,
                                *rgb.get((rgbx, rgby, 1)).unwrap() as u8,
                                *rgb.get((rgbx, rgby, 2)).unwrap() as u8,
                            )
                        } else {
                            (255, 255, 255)
                        };

                        let d = *depth.get((x, y)).unwrap();

                        points.push((
                            (y as i32) * 5,
                            (x as i32) * 5,
                            d as i32,
                            Color32::from_rgb(r, g, b),
                        ))
                    }

                    self.point_cloud.set_points(points);
                }
            }
        }

        self.point_cloud.render(ui);
    }

    fn context_menu(&mut self, ui: &mut Ui) {}
}

fn scale_shift_rgb_xy(x: usize, y: usize) -> Option<(usize, usize)> {
    static X_OFFSET: f32 = -10.;
    static Y_OFFSET: f32 = -10.;
    static SCAME: f32 = 1.9;

    let x = ((x as f32 + X_OFFSET) * SCAME) as usize;
    let y = ((y as f32 + Y_OFFSET) * SCAME) as usize;

    if x < 0 || y >= 640 {
        return None;
    }
    if y < 0 || x >= 480 {
        return None;
    }

    // println!("{}, {}", x, y);

    Some((x as usize, y as usize))
}

fn fetch_frame() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    is_success(&frame_config_encode(1, 0, 255, 0, 2, 7, 1, 0, 0))?;

    let url = format!("http://{}:{}/getdeep", HOST, PORT);

    let response = ureq::get(url).call()?;

    if response.status() != 200 {
        return Err(format!("Failed to get frame: HTTP {}", response.status()).into());
    }

    warn!("Got deep image");
    let deep_img = response.into_body().read_to_vec()?;
    warn!("Length={}", deep_img.len());

    // Parse frame ID and timestamp
    if deep_img.len() >= 16 {
        let mut cursor = Cursor::new(&deep_img[0..16]);
        let frame_id = cursor.read_u64::<LittleEndian>()?;
        let stamp_msec = cursor.read_u64::<LittleEndian>()?;
        warn!(
            "Frame ID: {}, Timestamp: {:.3}s",
            frame_id,
            stamp_msec as f64 / 1000.0
        );
    }

    return Ok(deep_img);
}

fn is_success(data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let url = format!("http://{}:{}/set_cfg", HOST, PORT);

    let response = ureq::post(url).send(data.to_vec())?;
    if response.status() == 200 {
        return Ok(());
    } else {
        return Err(format!("Status code: {}", response.status().to_string()).into());
    }
}

// Helper to display images in egui
fn image_widget(ui: &mut egui::Ui, id: &str, rgb_data: &[u8], size: [f32; 2]) {
    let color_image = egui::ColorImage::from_rgb([size[0] as usize, size[1] as usize], rgb_data);

    let handle = ui
        .ctx()
        .load_texture(id, color_image, egui::TextureOptions::LINEAR);

    let sized_image =
        egui::load::SizedTexture::new(handle.id(), egui::vec2(size[0] as f32, size[1] as f32));

    ui.image(sized_image);
}
