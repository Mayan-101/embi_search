use std::path::Path;
use std::time::Duration;

use embi_search::embedding::{self, EmbeddingEngine};
use embi_search::server::LlamaServer;
use embi_search::vectorstore::{DocumentChunk, VectorStore};

/// CLI test runner for Phases 1 & 2.
///
/// Phase 1: Spawn llama-server → embed two strings → print cosine similarity
/// Phase 2: Embed 5 documents → insert into LanceDB → query → assert top match
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llama_dir = Path::new("llama");
    let model_path = Path::new("nomic-embed-text-v1.5.Q4_K_M.gguf");
    let port: u16 = 8080;

    // ── Phase 1: Boot the embedding server ──────────────────────────────
    let server = LlamaServer::spawn(llama_dir, model_path, port)?;
    server.wait_until_ready(Duration::from_secs(60))?;

    let rt = tokio::runtime::Runtime::new()?;
    let engine = EmbeddingEngine::new(server.port());

    // ── Phase 1 Test: Cosine similarity ─────────────────────────────────
    {
        let text1 = "The quick brown fox jumps over the lazy dog";
        let text2 = "A fast dark colored canine leaps over an inactive hound";

        println!("\n=== Phase 1: Embedding Similarity Test ===");
        let vec1 = rt.block_on(engine.embed(text1))?;
        let vec2 = rt.block_on(engine.embed(text2))?;

        let similarity = embedding::cosine_similarity(&vec1, &vec2);
        println!("  Vector dimension: {}", vec1.len());
        println!("  Cosine Similarity: {:.6}", similarity);
        assert!(similarity > 0.5, "Similarity should be > 0.5 for related sentences");
        println!("  ✓ Phase 1 PASSED");
    }

    // ── Phase 2: Vector Store Integration Test ──────────────────────────
    {
        println!("\n=== Phase 2: Vector Store Integration Test ===");

        // Use a temporary database path for the test.
        let test_db_path = ".lancedb_test";
        if Path::new(test_db_path).exists() {
            std::fs::remove_dir_all(test_db_path)?;
        }

        let store = rt.block_on(VectorStore::connect(test_db_path))?;

        // 5 test documents with their real embeddings.
        let documents = vec![
            ("doc1", "C:\\docs\\rust_guide.txt",    "Rust is a systems programming language focused on safety and performance"),
            ("doc2", "C:\\docs\\python_intro.md",    "Python is a high-level interpreted language popular for data science"),
            ("doc3", "C:\\docs\\cooking_recipe.txt",  "Preheat the oven to 350°F and mix flour with sugar"),
            ("doc4", "C:\\docs\\travel_blog.md",      "The beaches of Bali are stunning with crystal clear water"),
            ("doc5", "C:\\docs\\rust_async.txt",      "Async programming in Rust uses futures and the tokio runtime"),
        ];

        println!("  Embedding {} documents...", documents.len());
        let mut chunks = Vec::new();
        for (id, path, content) in &documents {
            let vector = rt.block_on(engine.embed(content))?;
            chunks.push(DocumentChunk {
                id: id.to_string(),
                vector,
                file_path: path.to_string(),
                content_snippet: content.to_string(),
            });
        }

        // Insert all chunks.
        rt.block_on(store.insert_chunks(&chunks))?;

        let count = rt.block_on(store.count_rows())?;
        println!("  Rows in DB: {}", count);
        assert_eq!(count, 5, "Should have exactly 5 rows");

        // Query with the exact vector of doc1 ("Rust systems programming").
        println!("  Querying with doc1 vector...");
        let results = rt.block_on(store.search(&chunks[0].vector, 3))?;

        println!("  Top {} results:", results.len());
        for (i, r) in results.iter().enumerate() {
            println!("    {}. [dist={:.6}] {} — {}", i + 1, r.distance, r.file_path, r.content_snippet);
        }

        // Assert: top result should be doc1 itself (distance ≈ 0).
        assert_eq!(results[0].file_path, "C:\\docs\\rust_guide.txt",
            "Top result should be the queried document itself");
        assert!(results[0].distance < 1e-6,
            "Distance to self should be ~0, got {}", results[0].distance);

        // Assert: doc5 (Rust async) should be in top 3 (semantically similar to doc1).
        let top3_paths: Vec<&str> = results.iter().map(|r| r.file_path.as_str()).collect();
        assert!(top3_paths.contains(&"C:\\docs\\rust_async.txt"),
            "Rust async doc should be in top 3 similar to Rust guide");

        // Test delete.
        rt.block_on(store.delete_by_path("C:\\docs\\cooking_recipe.txt"))?;
        let count_after = rt.block_on(store.count_rows())?;
        assert_eq!(count_after, 4, "Should have 4 rows after deleting 1");

        println!("  ✓ Phase 2 PASSED");

        // Clean up test database.
        std::fs::remove_dir_all(test_db_path)?;
        println!("  Cleaned up test database");
    }

    println!("\n=== All tests passed! ===");
    println!("Server will be shut down.");

    // Runtime dropped, then server dropped (kills llama-server.exe).
    drop(rt);
    Ok(())
}