use std::{
    error::Error,
    sync::{Arc, Mutex},
    time::Instant,
};

use eframe::egui_glow;
use egui::{Align2, Color32, FontId, InputState, Stroke, Ui};
use glam::Vec3;
use wgpu::Color;

use crate::pane_manager::PsudoCreationContext;

use super::renderer::{self, Camera, PointRenderer};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct LivePointView {
    #[serde(skip)]
    renderer: Arc<Mutex<PointRenderer>>,
    #[serde(skip)]
    points: Vec<(i32, i32, i32, Color32)>,
}

impl Default for LivePointView {
    fn default() -> Self {
        Self {
            renderer: Arc::new(Mutex::new(PointRenderer::default())),
            points: Vec::new(),
        }
    }
}

impl LivePointView {
    pub fn init(&mut self, pcc: &PsudoCreationContext) {
        let renderer = self.renderer.lock();
        if renderer.is_ok() {
            renderer
                .unwrap()
                .init(pcc.gl.clone(), 100_000)
                .expect("Failed to init renderer!");
        } else {
            error!("Live point view - Failed to initilize renderer!");
        }
    }

    pub fn render(&mut self, ui: &mut Ui) {
        egui::Window::new("Live point view").show(ui.ctx(), |ui| {
            let max_rect = ui.max_rect();

            let renderer = self.renderer.clone();
            if renderer.lock().is_err() {
                error!("Live point view - Failed to lock renderer!");
                return;
            }
            if renderer.lock().unwrap().gl.is_none() {
                error!("Live point view - Failed to aquire opengl!");
                return;
            }
            renderer.lock().unwrap().clear();

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

            for &(x, y, z, color) in &self.points {
                renderer.lock().unwrap().add_point(x, y, z, color);
            }

            let o = <std::option::Option<Camera> as Clone>::clone(&renderer.lock().unwrap().camera)
                .unwrap()
                .orientation
                .clone();

            let cb = egui_glow::CallbackFn::new(move |_info, _painter| {
                renderer
                    .lock()
                    .unwrap()
                    .render(max_rect, input_state.clone());
            });

            let callback = egui::PaintCallback {
                rect: max_rect,
                callback: Arc::new(cb),
            };

            ui.painter().add(callback);
            let line_length: f32 = 20.;

            // if let Some(input_state) = input_state {
            //     if input_state.pointer.any_down() {

            let drawline = |pos: Vec3, color: Color32| {
                ui.painter().line_segment(
                    [
                        rect.center(),
                        rect.center()
                            + egui::Vec2 {
                                x: line_length * pos.x,
                                y: -line_length * pos.y,
                            },
                    ],
                    Stroke { width: 1.5, color },
                );
            };

            drawline(o.inverse() * glam::Vec3::X, Color32::RED);
            drawline(o.inverse() * glam::Vec3::Y, Color32::BLUE);
            drawline(o.inverse() * glam::Vec3::Z, Color32::GREEN);

            let text_size = 12.;

            ui.painter().text(
                max_rect.min,
                Align2::LEFT_TOP,
                format!("{} ms", start_time.elapsed().as_millis()),
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
        });
    }

    pub fn set_points(&mut self, points: Vec<(i32, i32, i32, Color32)>) {
        self.points = points;
    }
}
