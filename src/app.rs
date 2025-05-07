// use egui::{FontFamily, FontId, RichText, Visuals};
// use eframe::egui_glow;
// use std::{sync::Arc, time::Instant};
// use egui::{accesskit::TextAlign, mutex::Mutex, Align2, Color32, FontId, Pos2, Stroke};
// use egui_glow::glow;

use crate::pane_manager::PaneManager;

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
// #[derive(serde::Deserialize, serde::Serialize)]
// #[serde(default)]
pub struct App {
    pane_manager: PaneManager,
}

impl App {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Option<Self> {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        // if let Some(storage) = cc.storage {
        //     return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        // }

        Some(Self {
            pane_manager: PaneManager::new(cc),
        })
    }
}

// impl Default for App {
//     fn default() -> Self {
//         Self {
//             pane_manager: PaneManager::new(None),
//         }
//     }
// }

impl eframe::App for App {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.pane_manager.render(ui);
            // egui::scroll_area::ScrollArea::vertical().show(ui, |ui| {
            // egui::Frame::canvas(ui.style()).show(ui, |ui| {
            //     self.custom_painting(ui.max_rect(), ui);
            // });
            // })
        });
    }

    // fn on_exit(&mut self, gl: Option<&glow::Context>) {
    //     if let Some(gl) = gl {
    //         self.rotating_triangle.lock().destroy(gl);
    //     }
    // }

    // Called by the frame work to save state before shutdown.
    // fn save(&mut self, storage: &mut dyn eframe::Storage) {
    //     eframe::set_value(storage, eframe::APP_KEY, self);
    // }
}
