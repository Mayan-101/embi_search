use serde::Deserialize;
use std::path::PathBuf;

/// Path to the JSON configuration file (relative to the working directory).
const CONFIG_PATH: &str = "config.json";

/// Application-wide configuration loaded from `config.json` at startup.
///
/// Centralizes every value that was previously hardcoded across `main.rs`,
/// `embedding.rs`, and `vectorstore.rs`.
#[derive(Debug, Deserialize)]
pub struct AppConfig {
    /// Directory containing `llama-server.exe` and its DLLs.
    pub llama_dir: PathBuf,

    /// Path to the `.gguf` model file.
    pub model_path: PathBuf,

    /// TCP port for the llama-server HTTP endpoint.
    pub server_port: u16,

    /// Path to the LanceDB database directory.
    pub db_path: String,

    /// Directories to watch for file changes and index on startup.
    pub watch_dirs: Vec<PathBuf>,

    /// Dimensionality of the embedding vectors (e.g. 768 for Nomic Embed Text v1.5).
    pub vector_dim: i32,

    /// Model identifier sent to the `/v1/embeddings` endpoint.
    pub model_name: String,
}

impl AppConfig {
    /// Load configuration from `config.json` in the working directory.
    ///
    /// # Panics
    /// Panics with a descriptive message if the file is missing or malformed.
    pub fn load() -> Self {
        let content = std::fs::read_to_string(CONFIG_PATH)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", CONFIG_PATH, e));
        serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {}", CONFIG_PATH, e))
    }
}
