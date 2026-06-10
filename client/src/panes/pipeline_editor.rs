use crate::pane_manager::{Pane, PaneMode, PaneState, PsudoCreationContext};

use egui::Ui;
use egui::{Color32, Id, Pos2};
use egui_snarl::ui::WireStyle;
use egui_snarl::{
    InPin, NodeId, OutPin, OutPinId, Snarl,
    ui::{PinInfo, SnarlStyle, SnarlViewer},
};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct PipelinePane {
    snarl: Option<Snarl<Box<dyn Node>>>,
    style: Option<SnarlStyle>,
    snarl_ui_id: Option<Id>,
}
#[typetag::serde]
impl Pane for PipelinePane {
    fn new() -> PaneState
    where
        Self: Sized,
    {
        let mut s = Self {
            snarl: Some(Snarl::new()),
            style: Some(SnarlStyle::new()),
            snarl_ui_id: None,
        };
        PaneState {
            id: s.name().to_string(),
            mode: PaneMode::Hidden,
            pane: Box::new(s),
        }
    }
    fn init(&mut self, _cc: &PsudoCreationContext) {}
    fn name(&mut self) -> &str {
        "Pipeline Pane"
    }
    fn render(&mut self, ui: &mut Ui) {
        self.snarl_ui_id = Some(ui.id());

        if let Some(snarl) = &mut self.snarl {
            if let Some(style) = &self.style {
                snarl.show(&mut NodeViewer, style, "snarl", ui);
            }
        }
    }
    fn context_menu(&mut self, ui: &mut Ui) {
        ui.menu_button("Add Node", |ui| {
            if let Some(snarl) = &mut self.snarl {
                NodeViewer::add_node_menu(Pos2 { x: 0., y: 0. }, ui, snarl);
            }
        });
        if ui.button("Run").clicked() {
            ui.close_menu();
            self.run();
        }
        // if !self.snarl.is_none() {
        //     self.snarl.unwrap().add_node_menu(ui, ui.clip_rect().min.clone(), )
        // }
    }
}

impl PipelinePane {
    fn run(&mut self) {
        if let Some(snarl) = &mut self.snarl {
            fn remove_duplicates(nodes: Vec<NodeId>) -> Vec<NodeId> {
                let mut new_vec: Vec<NodeId> = Vec::new();
                for node in nodes {
                    if !new_vec.contains(&node) {
                        new_vec.push(node);
                    }
                }
                new_vec
            }
            fn has_input_wire(snarl: &Snarl<Box<dyn Node>>, nodeid: NodeId) -> bool {
                for wire in snarl.wires() {
                    if wire.1.node == nodeid {
                        return true;
                    }
                }
                false
            }
            fn get_output_wires(snarl: &Snarl<Box<dyn Node>>, nodeid: &NodeId) -> Vec<OutPinId> {
                let mut arr: Vec<OutPinId> = Vec::new();
                for wire in snarl.wires() {
                    if &wire.0.node == nodeid {
                        arr.push(wire.0)
                    }
                }
                arr
            }

            // let wires = snarl.wires().map(|| {})

            let mut nodes: Vec<Vec<NodeId>> = Vec::new();
            let mut starting_nodes: Vec<NodeId> = Vec::new();
            for node in snarl.nodes_ids_data() {
                if !has_input_wire(snarl, node.0) {
                    starting_nodes.push(node.0.clone())
                }
            }
            starting_nodes = remove_duplicates(starting_nodes);
            nodes.push(starting_nodes);

            for i in 1..50 {
                if nodes.get(i - 1).is_none() {
                    break;
                }
                let prevarr = nodes.get(i - 1).unwrap();
                if prevarr.len() == 0 {
                    break;
                }

                let mut newarr: Vec<NodeId> = Vec::new();

                for node in prevarr {
                    for wire in get_output_wires(snarl, node) {
                        newarr.push(wire.node);
                    }
                }

                newarr = remove_duplicates(newarr);

                nodes.push(newarr);
            }

            for nodearr in nodes {
                info!("Nodes: ");
                for node in nodearr {
                    info!("{}", snarl.get_node(node).unwrap().get_name());
                }
            }
        }
    }
}

fn _format_float(v: f64) -> String {
    let v = (v * 1000.0).round() / 1000.0;
    format!("{}", v)
}

#[typetag::serde(tag = "type")]
pub trait Node {
    fn new() -> Self
    where
        Self: Sized;
    fn get_name(&self) -> &str;
    fn get_description(&self) -> &str;
    fn duplicate(&self) -> Box<dyn Node>;
    fn inputs(&self) -> usize;
    fn outputs(&self) -> usize;
    fn show_input(&mut self, pin: &InPin, ui: &mut Ui, scale: f32) -> PinInfo;
    fn show_output(&mut self, pin: &OutPin, ui: &mut Ui, scale: f32) -> PinInfo;
    fn can_rx(&self, other: &Box<dyn Node>) -> bool;
    fn can_tx(&self, other: &Box<dyn Node>) -> bool;
    fn context_menu(&mut self, ui: &mut Ui);
    fn update(&mut self, ui: &mut Ui);
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Node1;
#[typetag::serde]
impl Node for Node1 {
    fn new() -> Self {
        Self
    }
    fn get_name(&self) -> &str {
        "Test"
    }
    fn get_description(&self) -> &str {
        "Test Node"
    }
    fn duplicate(&self) -> Box<dyn Node> {
        Box::new(Self::new())
    }
    fn inputs(&self) -> usize {
        1
    }
    fn outputs(&self) -> usize {
        1
    }
    fn show_input(&mut self, _pin: &InPin, _ui: &mut Ui, _scale: f32) -> PinInfo {
        PinInfo::square()
    }
    fn show_output(&mut self, _pin: &OutPin, _ui: &mut Ui, _scale: f32) -> PinInfo {
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
        ui.label("Test!");
    }
    fn update(&mut self, _ui: &mut Ui) {}
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Node2;
#[typetag::serde]
impl Node for Node2 {
    fn new() -> Self {
        Self
    }
    fn get_name(&self) -> &str {
        "Test 2-1"
    }
    fn get_description(&self) -> &str {
        "Test Node"
    }
    fn duplicate(&self) -> Box<dyn Node> {
        Box::new(Self::new())
    }
    fn inputs(&self) -> usize {
        2
    }
    fn outputs(&self) -> usize {
        1
    }
    fn show_input(&mut self, _pin: &InPin, _ui: &mut Ui, _scale: f32) -> PinInfo {
        PinInfo::square()
    }
    fn show_output(&mut self, _pin: &OutPin, _ui: &mut Ui, _scale: f32) -> PinInfo {
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
        ui.label("Test!");
    }
    fn update(&mut self, _ui: &mut Ui) {}
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Node3;
#[typetag::serde]
impl Node for crate::panes::pipeline_editor::Node3 {
    fn new() -> Self {
        Self
    }
    fn get_name(&self) -> &str {
        "Test 1-2"
    }
    fn get_description(&self) -> &str {
        "Test Node"
    }
    fn duplicate(&self) -> Box<dyn Node> {
        Box::new(Self::new())
    }
    fn inputs(&self) -> usize {
        1
    }
    fn outputs(&self) -> usize {
        2
    }
    fn show_input(&mut self, _pin: &InPin, _ui: &mut Ui, _scale: f32) -> PinInfo {
        PinInfo::square()
    }
    fn show_output(&mut self, _pin: &OutPin, _ui: &mut Ui, _scale: f32) -> PinInfo {
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
        ui.label("Test!");
    }
    fn update(&mut self, _ui: &mut Ui) {}
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct NodeViewer;

impl SnarlViewer<Box<dyn Node>> for NodeViewer {
    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<Box<dyn Node>>) {
        // Validate connection

        let rx = snarl.get_node(to.id.node).unwrap();
        let tx = snarl.get_node(from.id.node).unwrap();

        if rx.can_rx(tx) && tx.can_tx(rx) {
            for &remote in &to.remotes {
                snarl.disconnect(remote, to.id);
            }

            snarl.connect(from.id, to.id);
        }
    }

    fn disconnect(&mut self, _from: &OutPin, to: &InPin, snarl: &mut Snarl<Box<dyn Node>>) {
        for &remote in &to.remotes {
            snarl.disconnect(remote, to.id);
        }
    }

    fn title(&mut self, node: &Box<dyn Node>) -> String {
        node.get_name().to_string()
    }

    fn outputs(&mut self, node: &Box<dyn Node>) -> usize {
        node.outputs()
    }

    fn inputs(&mut self, node: &Box<dyn Node>) -> usize {
        node.inputs()
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut Ui,
        scale: f32,
        snarl: &mut Snarl<Box<dyn Node>>,
    ) -> PinInfo {
        snarl
            .get_node_mut(pin.id.node)
            .unwrap()
            .show_input(pin, ui, scale)
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut Ui,
        scale: f32,
        snarl: &mut Snarl<Box<dyn Node>>,
    ) -> PinInfo {
        snarl
            .get_node_mut(pin.id.node)
            .unwrap()
            .show_output(pin, ui, scale)
    }

    fn has_graph_menu(&mut self, _pos: Pos2, _snarl: &mut Snarl<Box<dyn Node>>) -> bool {
        true
    }

    fn show_graph_menu(
        &mut self,
        pos: egui::Pos2,
        ui: &mut Ui,
        _scale: f32,
        snarl: &mut Snarl<Box<dyn Node>>,
    ) {
        NodeViewer::add_node_menu(pos, ui, snarl);
    }

    fn has_on_hover_popup(&mut self, _: &Box<dyn Node>) -> bool {
        true
    }

    fn show_on_hover_popup(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        _scale: f32,
        snarl: &mut Snarl<Box<dyn Node>>,
    ) {
        ui.label(snarl.get_node(node).unwrap().get_description());
    }

    fn has_node_menu(&mut self, _node: &Box<dyn Node>) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        nodeid: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        _scale: f32,
        snarl: &mut Snarl<Box<dyn Node + 'static>>,
    ) {
        ui.label("Node menu");
        snarl.get_node_mut(nodeid).unwrap().context_menu(ui);
        if ui.button("Remove").clicked() {
            snarl.remove_node(nodeid);
            ui.close_menu();
        } else if ui.button("Duplicate").clicked() {
            let node = snarl.get_node_mut(nodeid).unwrap().duplicate();
            snarl.insert_node(egui::pos2(0., 0.), node);
            ui.close_menu();
            // }// else if ui.button("Remove All Connections").clicked() {
            //     ui.
            //     ui.close_menu();
        }
    }

    fn has_body(&mut self, _node: &Box<dyn Node>) -> bool {
        true
    }

    fn show_body(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        _scale: f32,
        snarl: &mut Snarl<Box<dyn Node>>,
    ) {
        snarl.get_node_mut(node).unwrap().update(ui);
    }
}

impl NodeViewer {
    pub fn add_node_menu(pos: Pos2, ui: &mut Ui, snarl: &mut Snarl<Box<dyn Node>>) {
        ui.label("Add node");

        if ui.button("Test").clicked() {
            snarl.insert_node(pos, Box::new(Node1::new()));
            ui.close_menu();
        } else if ui.button("Constants").clicked() {
            snarl.insert_node(pos, Box::new(crate::nodes::constants::Constants::new()));
            ui.close_menu();
        } else if ui.button("Test 2-1").clicked() {
            snarl.insert_node(pos, Box::new(Node2::new()));
            ui.close_menu();
        } else if ui.button("Test 1-2").clicked() {
            snarl.insert_node(pos, Box::new(Node3::new()));
            ui.close_menu();
        }
    }
}
