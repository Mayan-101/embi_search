use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

/// Manages the lifecycle of a `llama-server.exe` child process.
///
/// The server is spawned as a hidden background process (no console window on
/// Windows) and is automatically killed when this struct is dropped.
pub struct LlamaServer {
    child: Child,
    port: u16,
}

impl LlamaServer {
    /// Spawn `llama-server.exe` with the given GGUF model, bound to `127.0.0.1:<port>`.
    ///
    /// The server is started with `--embedding` mode enabled and a context size
    /// of 8192 tokens (suitable for Nomic Embed Text v1.5).
    ///
    /// # Arguments
    /// * `llama_dir` — directory containing `llama-server.exe` and its DLLs
    /// * `model_path` — path to the `.gguf` model file
    /// * `port` — TCP port to bind the HTTP server to
    pub fn spawn(
        llama_dir: &Path,
        model_path: &Path,
        port: u16,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let exe = llama_dir.join("llama-server.exe");

        if !exe.exists() {
            return Err(format!("llama-server.exe not found at {}", exe.display()).into());
        }
        if !model_path.exists() {
            return Err(format!("Model file not found at {}", model_path.display()).into());
        }

        let mut cmd = Command::new(&exe);

        cmd.args([
            "--model",
            &model_path.to_string_lossy(),
            "--port",
            &port.to_string(),
            "--host",
            "127.0.0.1",
            "--embedding",
            "--ctx-size",
            "8192",
        ]);

        // On Windows, hide the console window for the child process.
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let child = cmd.spawn()?;

        println!(
            "[server] Spawned llama-server.exe (PID {}) on port {}",
            child.id(),
            port
        );

        Ok(Self { child, port })
    }

    /// Block until the server responds with HTTP 200 on `/health`, or timeout.
    ///
    /// This uses a blocking reqwest client internally because it is called
    /// exactly once at startup before the async runtime is in full swing.
    pub fn wait_until_ready(&self, timeout: Duration) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("http://127.0.0.1:{}/health", self.port);
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()?;

        let start = Instant::now();
        let poll_interval = Duration::from_millis(500);

        println!("[server] Waiting for llama-server to become ready...");

        loop {
            if start.elapsed() > timeout {
                return Err(format!(
                    "llama-server did not become ready within {:.0}s",
                    timeout.as_secs_f64()
                )
                .into());
            }

            match client.get(&url).send() {
                Ok(resp) if resp.status().is_success() => {
                    println!(
                        "[server] llama-server ready after {:.1}s",
                        start.elapsed().as_secs_f64()
                    );
                    return Ok(());
                }
                _ => {
                    std::thread::sleep(poll_interval);
                }
            }
        }
    }

    /// Return the port this server is bound to.
    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for LlamaServer {
    /// Kill the child process when the `LlamaServer` is dropped.
    /// This prevents zombie `llama-server.exe` processes.
    fn drop(&mut self) {
        let pid = self.child.id();
        match self.child.kill() {
            Ok(()) => println!("[server] Killed llama-server.exe (PID {})", pid),
            Err(e) => eprintln!("[server] Failed to kill llama-server.exe (PID {}): {}", pid, e),
        }
        // Reap the child to avoid zombie processes.
        let _ = self.child.wait();
    }
}
