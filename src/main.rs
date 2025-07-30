//! Sarissa CLI binary.

use std::io::Write;
use std::process;

use clap::Parser;
use env_logger::Builder;
use log::LevelFilter;

use sarissa::cli::args::*;
use sarissa::cli::commands::*;

fn main() {
    // Parse command line arguments using clap
    let args = SarissaArgs::parse();

    // Set up logging/verbosity based on args if needed
    let log_level = match args.verbosity() {
        0 => LevelFilter::Error, // Quiet mode
        1 => LevelFilter::Warn,  // Default
        2 => LevelFilter::Info,  // Verbose
        _ => LevelFilter::Debug, // Very verbose (3+)
    };

    Builder::new()
        .filter_level(log_level)
        .format(|buf, record| writeln!(buf, "[{}] {}", record.level(), record.args()))
        .init();

    // Execute the command
    if let Err(e) = execute_command(args) {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}
