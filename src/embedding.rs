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
    /// Create a new engine pointing at the given port on localhost.
    pub fn new(port: u16) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: format!("http://127.0.0.1:{}/v1/embeddings", port),
            model: "nomic-embed-text-v1.5".to_string(),
        }
    }

    /// Send `text` to the embedding server and return the 768-dimensional vector.
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
            .await?
            .json::<EmbeddingResponse>()
            .await?;

        // Nomic Embed Text v1.5 outputs a 768-dimensional vector.
        Ok(response.data[0].embedding.clone())
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns a value in [-1, 1] where 1 means identical direction.
/// Panics if the vectors have different lengths.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors should have similarity 1.0");
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal vectors should have similarity 0.0");
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6, "opposite vectors should have similarity -1.0");
    }
}
