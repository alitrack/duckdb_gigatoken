# duckdb_gigatoken

DuckDB extension for fast BPE tokenization using [Gigatoken](https://github.com/marcelroed/gigatoken) — the world's fastest tokenizer (989× faster than HuggingFace tokenizers on GPT-2).

**Tokenize text at GB/s, directly in DuckDB. No Python required.**

## Quickstart

```sql
LOAD 'gigatoken.duckdb_extension';

-- Single text
SELECT * FROM gigatoken_encode('/path/to/tokenizer.json', 'Hello world!');
-- → (idx INT, token_id INT)

-- Batch from table
SELECT d.id, t.idx, t.token_id
FROM documents d, gigatoken_encode('/path/to/tokenizer.json', d.content) t;

-- Decode
SELECT text FROM gigatoken_decode('/path/to/tokenizer.json', '[15496, 995, 0]');
-- → 'Hello world!'
```

## SQL API

| Function | Parameters | Returns |
|---|---|---|
| `gigatoken_encode(path, text)` | tokenizer.json path, text | (idx, token_id) |
| `gigatoken_encode_batch(path, texts)` | path, JSON array of strings | (doc, idx, token_id) |

## Installation

Download from [Releases](https://github.com/alitrack/duckdb_gigatoken/releases) or build from source:

```bash
git clone git@github.com:alitrack/duckdb_gigatoken.git && cd duckdb_gigatoken
cargo build --release
python3 scripts/metadata.py target/release/libgigatoken.so -o gigatoken.duckdb_extension
```

Requires **nightly Rust** (gigatoken uses `portable_simd`).

## Supported Tokenizers

Any HuggingFace-compatible BPE `tokenizer.json` file (GPT-2, Llama, Qwen, DeepSeek, GLM, Gemma, etc.). See the [Gigatoken README](https://github.com/marcelroed/gigatoken#benchmarks) for the full list.

## Architecture

```
DuckDB SQL
  └─ Table Function (VTab)
       └─ gigatoken Rust core
            ├─ SIMD pretokenization (AVX-512/AVX2/NEON)
            ├─ ShortPretokenCache (2 MiB huge-page open-addressing table)
            └─ PairRankTable (dense + flat merge lookup)
```

Tokenizer is loaded once and forked per call — immutable tables are shared behind `Arc`, caches warm up during queries.

## License

MIT — matches gigatoken (MIT) and duckdb-rs (MIT).
