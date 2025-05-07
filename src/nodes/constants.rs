// use crate::pane_manager::{Pane, PaneMode, PaneState, PsudoCreationContext};
use crate::panes::pipeline_editor::Node;
use eframe::epaint::Color32;
use egui::{Id, Ui};
use egui_snarl::ui::{PinInfo, WireStyle};
use egui_snarl::{InPin, OutPin};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Constants {
    vars: Vec<(String, String)>,
    popup_open: bool,
    uid: Id,
}
#[typetag::serde]
impl Node for Constants {
    fn new() -> Self {
        let mut s = Self {
            vars: Vec::new(),
            popup_open: false,
            uid: Id::new(rand::random::<u64>()),
        };
        s.vars.push(("VAR".to_string(), "Change Me".to_string()));
        s
    }

    fn get_name(&self) -> &str {
        "Constants"
    }
    fn get_description(&self) -> &str {
        "Test Node"
    }

    fn duplicate(&self) -> Box<dyn Node> {
        Box::new(Self::new())
    }

    fn inputs(&self) -> usize {
        0
    }
    fn outputs(&self) -> usize {
        self.vars.len()
    }
    fn show_input(&mut self, _pin: &InPin, _ui: &mut Ui, _scale: f32) -> PinInfo {
        PinInfo::square()
    }
    fn show_output(&mut self, pin: &OutPin, ui: &mut Ui, _scale: f32) -> PinInfo {
        ui.label(self.vars.iter().nth(pin.id.output).unwrap().0.clone());
        PinInfo::square()
            .with_fill(Color32::RED)
            .with_wire_style(WireStyle::Bezier3)
    }
    fn can_rx(&self, _other: &Box<dyn Node>) -> bool {
        true
    }
    fn can_tx(&self, _other: &Box<dyn Node>) -> bool {
        true
    }
    fn context_menu(&mut self, ui: &mut Ui) {
        // egui::Window::new("TEST").show(ui.ctx(), |ui| {
        //     ui.heading("EEEEE");
        // });
        if ui.button("Edit").clicked() {
            self.popup_open = !self.popup_open;
            ui.close_menu();
        }
    }
    fn update(&mut self, ui: &mut Ui) {
        if self.popup_open {
            egui::Window::new("Edit - ".to_owned() + self.get_name())
                .id(self.uid)
                .show(ui.ctx(), |ui| {
                    egui::Grid::new("my_grid")
                        .striped(true)
                        .max_col_width(9999.)
                        .show(ui, |ui| {
                            for (_i, (var1, _var2)) in self.vars.iter().enumerate() {
                                if var1.is_empty() {
                                    // self.vars.remove()
                                }
                                // ui.add(egui::TextEdit::singleline(var1.as_mut()));
                                // ui.add(egui::TextEdit::singleline(var2.as_mut()));
                                ui.end_row();
                            }
                        });
                    if ui.button("ADD").clicked() {
                        self.vars.push(("VAR".to_string(), "Change Me".to_string()));
                    }
                });
        }
    }
}

// #[derive(serde::Serialize, serde::Deserialize, Clone)]
// pub struct Constant_Edit_Popup {
//     pub data: Vec<String>,
//     pub has_changed: bool,
// }
// #[typetag::serde]
// impl Pane for Constant_Edit_Popup {
//     fn new() -> PaneState where Self: Sized {
//         let mut s = Self {
//             node: None,
//             has_changed: false,
//         };
//         PaneState {
//             id: s.name().to_string(),
//             mode: PaneMode::Popup,
//             pane: Box::new(s),
//         }
//     }
//     fn init(&mut self, _pcc: &PsudoCreationContext) {}
//     fn name(&mut self) -> &str {"ERROR"}
//     fn render(&mut self, _ui: &mut Ui){}
//     fn context_menu(&mut self, _ui: &mut Ui) {}
// }
