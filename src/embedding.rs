use serde::{Deserialize, Serialize};

/// Request body for the llama.cpp `/v1/embeddings` endpoint.
#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    input: &'a str,
    model: &'a str,
}

/// Top-level response from the embeddings endpoint.
#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

/// Individual embedding entry within the response.
#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Async HTTP client for generating text embeddings via a local llama.cpp server.
///
/// Wraps a `reqwest::Client` and targets the OpenAI-compatible `/v1/embeddings`
/// endpoint exposed by `llama-server.exe --embedding`.
pub struct EmbeddingEngine {
    client: reqwest::Client,
    url: String,
    model: String,
}

impl EmbeddingEngine {
    /// Create a new engine pointing at the given port on localhost,
    /// using the specified model name for the embedding requests.
    pub fn new(port: u16, model_name: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: format!("http://127.0.0.1:{}/v1/embeddings", port),
            model: model_name.to_string(),
        }
    }

    /// Send `text` to the embedding server and return the embedding vector.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let req_body = EmbeddingRequest {
            input: text,
            model: &self.model,
        };

        let response = self
            .client
            .post(&self.url)
            .json(&req_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("Llama server error ({}): {}", status, error_text).into());
        }

        let parsed = response.json::<EmbeddingResponse>().await?;

        Ok(parsed.data[0].embedding.clone())
    }
}
