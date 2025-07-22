//! Sarissa CLI binary.

use clap::Parser;
use sarissa::cli::{args::*, commands::*};
use std::process;

fn main() {
    // Parse command line arguments using clap
    let args = SarissaArgs::parse();

    // Set up logging/verbosity based on args if needed
    if args.verbosity() >= 3 {
        // Enable debug logging
        unsafe {
            std::env::set_var("RUST_LOG", "debug");
        }
    } else if args.verbosity() >= 2 {
        // Enable verbose logging
        unsafe {
            std::env::set_var("RUST_LOG", "info");
        }
    } else if args.verbosity() == 0 {
        // Quiet mode
        unsafe {
            std::env::set_var("RUST_LOG", "error");
        }
    }

    // Execute the command
    if let Err(e) = execute_command(args) {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}
