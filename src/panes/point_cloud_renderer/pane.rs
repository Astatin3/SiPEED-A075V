use std::{sync::Arc, time::Instant};

use eframe::egui_glow;
use egui::{Align2, Color32, FontId, InputState, Stroke, Ui, mutex::Mutex};

use crate::pane_manager::{Pane, PaneMode, PaneState, PsudoCreationContext};

use super::renderer::{Camera, PointRenderer};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct PointRendererPane {
    #[serde(skip)]
    renderer: Arc<Mutex<PointRenderer>>,
    #[serde(skip)]
    points: Vec<(i32, i32, i32, Color32)>,
    #[serde(skip)]
    file_dialog_open: bool,
    #[serde(skip)]
    cur_path: String,
}

#[typetag::serde]
impl Pane for PointRendererPane {
    fn new() -> PaneState
    where
        Self: Sized,
    {
        let renderer = PointRenderer::default();
        let mut s = Self {
            renderer: Arc::new(Mutex::new(renderer)),
            points: Vec::new(),
            file_dialog_open: false,
            cur_path: "./".to_string(),
        };
        PaneState {
            id: s.name().to_string(),
            mode: PaneMode::Hidden,
            pane: Box::new(s),
        }
    }
    fn init(&mut self, pcc: &PsudoCreationContext) {
        self.renderer.lock().init(pcc.gl.clone(), 1_000_000);
    }
    fn name(&mut self) -> &str {
        "Point Cloud"
    }
    fn render(&mut self, ui: &mut Ui) {
        let max_rect = ui.max_rect();

        let renderer = self.renderer.clone();
        if renderer.lock().gl.is_none() {
            return;
            // renderer.lock().expect("Renderer Not Initialized").init(ui.ctx()., 1_000_000);
        }
        renderer.lock().clear();

        if self.file_dialog_open {
            egui::Window::new("Load PLY File").show(ui.ctx(), |ui| {
                ui.label("Enter PLY file path:");
                ui.text_edit_singleline(&mut self.cur_path); // Add proper path handling

                ui.horizontal(|ui| {
                    if ui.button("Load").clicked() {
                        let renderer = &mut renderer.lock();
                        // Add proper path handling and error reporting
                        let ply = renderer.load_ply();
                        if let Err(e) = ply {
                            warn!("Failed to load PLY: {}", e);
                        } else {
                            // self.renderer.lock().camera.reset();
                            self.points = ply.unwrap();
                        }

                        self.file_dialog_open = false;
                    }
                    if ui.button("Cancel").clicked() {
                        self.file_dialog_open = false;
                    }
                });
            });
        }

        let start_time = Instant::now();

        let (rect, response) = ui.allocate_exact_size(
            egui::Vec2 {
                x: max_rect.width(),
                y: max_rect.height(),
            },
            egui::Sense::drag(),
        );

        let input_state: Option<InputState> = ui.input(|input_state| {
            if response.hovered() {
                //&& response.has_focus() {
                Some(input_state.clone())
            } else {
                None
            }
        });

        if self.points.is_empty() {
            let radius = 1000i32;
            for i in 0..100000 {
                //    let theta = (i as f32 * 0.1).sin() * std::f32::consts::PI;
                //    let phi = (i as f32 * 0.1).cos() * std::f32::consts::PI;

                let x = (radius as f32 * (i as f32).cos()) as i32;
                let y = (radius as f32 * (i as f32).sin()) as i32;
                let z = (i as f32 * 0.05) as i32;

                // let x = (i as f32 * 0.1) as u32;
                // let y = (i as f32 * 0.1) as u32 ;
                // let z = (i as f32 * 0.1) as u32;

                // Color based on position
                let color = Color32::from_rgba_premultiplied(
                    ((x as f32 / radius as f32) * 255.0) as u8,
                    ((y as f32 / radius as f32) * 255.0) as u8,
                    ((z as f32 / radius as f32) * 255.0) as u8,
                    255,
                );

                self.points.push((x, y, z, color));
            }
        }

        // let painter = ui.painter();

        for &(x, y, z, color) in &self.points {
            renderer.lock().add_point(x, y, z, color);
        }

        let o = <std::option::Option<Camera> as Clone>::clone(&renderer.lock().camera)
            .unwrap()
            .orientation
            .clone();

        let cb = egui_glow::CallbackFn::new(move |_info, _painter| {
            renderer.lock().render(max_rect, input_state.clone());
        });

        let callback = egui::PaintCallback {
            rect: max_rect,
            callback: Arc::new(cb),
        };

        ui.painter().add(callback);
        let line_length: f32 = 20.;

        // if let Some(input_state) = input_state {
        //     if input_state.pointer.any_down() {

        let pos1 = o.inverse() * glam::Vec3::X;
        let pos2 = o.inverse() * glam::Vec3::Y;
        let pos3 = o.inverse() * glam::Vec3::Z;

        ui.painter().line_segment(
            [
                rect.center(),
                rect.center()
                    + egui::Vec2 {
                        x: line_length * pos1.x,
                        y: -line_length * pos1.y,
                    },
            ],
            Stroke {
                width: 1.5,
                color: Color32::RED,
            },
        );

        ui.painter().line_segment(
            [
                rect.center(),
                rect.center()
                    + egui::Vec2 {
                        x: line_length * pos2.x,
                        y: -line_length * pos2.y,
                    },
            ],
            Stroke {
                width: 1.5,
                color: Color32::BLUE,
            },
        );

        ui.painter().line_segment(
            [
                rect.center(),
                rect.center()
                    + egui::Vec2 {
                        x: line_length * pos3.x,
                        y: -line_length * pos3.y,
                    },
            ],
            Stroke {
                width: 1.5,
                color: Color32::GREEN,
            },
        );
        // }}

        let end_time = Instant::now();

        // println!("{}", end_time.duration_since(start_time).as_millis());

        let text_size = 12.;

        ui.painter().text(
            max_rect.min,
            Align2::LEFT_TOP,
            format!("{} ms", end_time.duration_since(start_time).as_millis()),
            FontId::monospace(text_size),
            Color32::WHITE,
        );

        ui.painter().text(
            max_rect.min
                + egui::Vec2 {
                    x: 0.,
                    y: text_size,
                },
            Align2::LEFT_TOP,
            format!("{} points", self.points.len()),
            FontId::monospace(text_size),
            Color32::WHITE,
        );
    }
    fn context_menu(&mut self, ui: &mut Ui) {
        if ui.button("Load PLY").clicked() {
            self.file_dialog_open = true;
        }
    }
}
