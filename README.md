# embi_search 

> **A lightning-fast, privacy-first local semantic file search engine for Windows — built as a lightweight alternative to the native OS search overlay. (Basically, FUCK WINDOWS NATIVE SEARCH, Running an Embedding model in the background is somehow still faster then that hell of a mess)**

`embi_search` replaces the clunky Windows search experience with a floating, keyboard-driven overlay that understands *meaning*, not just keywords. Under the hood, a Rust binary manages the full lifecycle: spawning a local `llama.cpp` inference server, harvesting and watching your filesystem, chunking and embedding file content, and persisting those vectors into a local LanceDB database — all without a single byte leaving your machine.


---

## Features

- **Semantic search** — find files by intent and meaning, not just filename or keywords
- **Fully offline & private** — no cloud APIs, no telemetry; all inference runs on-device via `llama.cpp`
- **Real-time indexing** — OS-level filesystem watcher re-embeds files the moment they are created or modified
- **Sub-millisecond retrieval** — nearest-neighbor search over a persistent LanceDB vector index
- **Minimal UI** — a frameless, transparent Tauri overlay activated with `Alt+Space`, dismissed on focus loss
- **Native file opening** — click any result to open it with its default OS handler


---

## Getting Started

### Prerequisites

- **Rust** (stable, 2021 edition) — [rustup.rs](https://rustup.rs)
- **Tauri CLI** — `cargo install tauri-cli`
- **llama.cpp pre-built binaries** — download `llama-server.exe` and required DLLs from the [llama.cpp releases page](https://github.com/ggml-org/llama.cpp/releases) and place them in the `llama/` directory.
- **Nomic Embed Text v1.5 GGUF model** — download `nomic-embed-text-v1.5.Q4_K_M.gguf` from [Hugging Face](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) and place it in the `model/` directory.

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/embi_search.git
cd embi_search

# Place llama-server.exe + DLLs into ./llama/
# Place nomic-embed-text-v1.5.Q4_K_M.gguf into ./model/

# Verify structure
ls llama/   # should contain llama-server.exe
ls model/   # should contain *.gguf
```

### Running

```bash
# Development build (with Tauri DevTools)
cargo tauri dev

# Production build (optimized for size: opt-level=z, LTO, stripped symbols)
cargo tauri build
```

The release profile is tuned for minimal binary size: `opt-level = "z"`, `lto = true`, `strip = true`.

---

## Supported File Types

| Extension | Type            |
|-----------|-----------------|
| `.txt`    | Plain text      |
| `.md`     | Markdown        |
| `.rs`     | Rust source     |
| `.py`     | Python source   |
| `.csv`    | Comma-separated |
| `.json`   | JSON data       |
| `.toml`   | TOML config     |

---

## Usage

1. Launch the application. The overlay starts hidden.
2. Press **`Alt+Space`** to summon the search bar.
3. Type a natural language query — e.g. *"notes about rotating black holes"* or *"database migration script"*.
4. Results appear as ranked file matches with a content snippet. Click any result to open it natively.
5. The window auto-dismisses when it loses focus. Press `Alt+Space` again to toggle.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tauri Application                        │
│                                                                 │
│   ┌───────────────┐    IPC (invoke)    ┌───────────────────┐    │
│   │  dist/        │ ◄────────────────► │  src/main.rs      │    │
│   │  index.html   │  search_files()    │  (AppState, cmds) │    │
│   │  (WebView UI) │  open_file()       └────────┬──────────┘    │
│   └───────────────┘                             │               │
│                                                 │               │
│              ┌──────────────────────────────────┤               │
│              │               │                  │               │
│   ┌──────────▼──────┐  ┌─────▼──────┐  ┌────────▼──────────┐    │
│   │ embedding.rs    │  │ harvester  │  │  vectorstore.rs   │    │
│   │                 │  │    .rs     │  │                   │    │
│   │ EmbeddingEngine │  │ Directory  │  │  VectorStore      │    │
│   │ (reqwest HTTP   │  │ Harvester  │  │  (LanceDB, Arrow) │    │
│   │  client)        │  │ + Watcher  │  │                   │    │
│   └────────┬────────┘  └─────┬──────┘  └───────────────────┘    │
│            │                 │                   ▲              │
│            │           FileEvent (mpsc)          │              │
│            │           Upsert / Remove           │ insert/      │
│            │                 │                   │ delete/      │
│            │           ┌─────▼──────────────┐    │ search       │
│            │           │  Background Worker │────┘              │
│            │           │  (async Tokio task)│                   │
│            │           └────────────────────┘                   │
│            │                                                    │
│   ┌────────▼────────────────────────────────────┐               │
│   │               server.rs                     │               │
│   │  LlamaServer — spawns llama-server.exe as   │               │
│   │  a hidden child process; killed on Drop     │               │
│   └────────┬────────────────────────────────────┘               │
└────────────┼────────────────────────────────────────────────────┘
             │  HTTP  POST /v1/embeddings
             │  (OpenAI-compatible API, localhost:8080)
   ┌─────────▼-──────────────────────────────────────┐
   │            llama-server.exe (llama.cpp)         │
   │  Model: nomic-embed-text-v1.5.Q4_K_M.gguf       │
   │  Mode:  --embedding  --ctx-size 8192            │
   │  Bind:  127.0.0.1:8080                          │
   └─────────────────────────────────────────────────┘
```

---
## Future Plans

### PDF & Image Support via Multimodal Indexing

The next major milestone is extending `embi_search` beyond plaintext into the two most common document types on a typical machine: PDFs and images.

**PDF Indexing with OCR**

For PDFs with embedded selectable text, `pdf-extract` (already included in `Cargo.toml`) can extract raw text directly, which is then fed into the existing embedding pipeline unchanged. For scanned PDFs or image-only documents, the plan is to integrate a lightweight **OCR** stage using a local model such as [Tesseract](https://github.com/tesseract-ocr/tesseract) (via `leptess` Rust bindings) or the ONNX-exported version of **PaddleOCR**. The extracted text is stored as the `content_snippet` in LanceDB, making scanned documents fully searchable by meaning.

**Image Indexing with Vision-Language Models**

For images (`.png`, `.jpg`, `.webp`, etc.), raw pixel data cannot be directly embedded by a text model. Instead, the plan is to introduce a **Vision Transformer (ViT) captioning stage**: a local vision-language model such as [MoonDream](https://github.com/vikhyat/moondream) or a GGUF-quantized **LLaVA** variant running through the same `llama.cpp` server (which supports multimodal inference) would generate a natural language description of each image. That generated description is then passed through the existing `EmbeddingEngine` and stored in the vector store like any other document chunk.

**Proposed data flow for images:**

```
image file detected by watcher
        │
  ViT / VLM model (MoonDream or LLaVA via llama.cpp multimodal)
        │
  generated caption / description (text)
        │
  EmbeddingEngine.embed(caption) → Vec<f32>
        │
  VectorStore.insert_chunks([{
      id:              uuid,
      vector:          embedding,
      file_path:       "C:/Users/.../photo.jpg",
      content_snippet: "A sunset over a mountain range with orange clouds..."
  }])
```

This approach means the same `VectorStore`, `EmbeddingEngine`, and search path are reused without modification — only the content extraction stage changes per file type. The schema in LanceDB may be extended with a `source_type` column (`text`, `pdf`, `image`) to enable filtered queries in future versions.

---

*Built with Rust, Tauri, llama.cpp, and LanceDB.*
