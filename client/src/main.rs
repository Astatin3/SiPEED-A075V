use SiPEED_A075V::app::App;

// Main function
fn main() -> Result<(), eframe::Error> {
    env_logger::init();

    let options = eframe::NativeOptions {
        // viewport: egui::ViewportBuilder::default()
        //     .with_inner_size([400.0, 300.0])
        //     .with_min_inner_size([300.0, 220.0]),
        depth_buffer: 24,
        ..Default::default()
    };

    eframe::run_native(
        "Frame Viewer",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc).unwrap()))),
    )
}
