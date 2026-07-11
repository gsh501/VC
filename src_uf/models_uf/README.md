# DCVC-UF Model

This folder contains an isolated DCVC-UF style implementation. The original `src/models` RT implementation is untouched.

Implemented UF structural pieces:

- chunk-based coding with configurable `chunk_size`, defaulting to 8 for high-throughput UF
- one compact latent for a whole chunk instead of one latent per frame
- cross-frame interaction through patchified chunk features and a shared chunk encoder/decoder
- cross-chunk context propagation through the decoded chunk feature state
- parallel frame-specific decoders for each temporal position in the chunk
- streamlined entropy model: scales for all four quadtree-like partitions are estimated once, while means are restored progressively without extra bitstream interactions

Model presets from the UF paper:

| preset | chunk size | chunk encoder DC blocks | chunk decoder DC blocks | context DC blocks | per-frame decoder DC blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ht-s` | 8 | 6 | 7 | 11 | 3 |
| `ht-l` | 8 | 7 | 11 | 12 | 5 |
| `ld` | 1 | 3 | 3 | 9 | 3 |

Main exports:

- `DCVCUF`: UF video model, compatible alias `DMC`
- `DCVCUFIntra`: intra model alias `DMCI`
- `UF_MODEL_CONFIGS` / `get_uf_model_config`: preset metadata

Useful entry points:

- `forward(chunk, qp)` for one chunk shaped `B,T,3,H,W`
- `forward_sequence(sequence, qp)` for non-overlapping chunk processing with reference propagation
- `compress(chunk, qp)` / `decompress(bit_stream, sps, qp)` for one chunk
- `compress_sequence(sequence, qp)` / `decompress_sequence(packets, qp)` for chunk-by-chunk coding

The structure now follows UF rather than RT's frame-by-frame backbone. Paper-level rate-distortion performance still requires UF training, UF checkpoints, and evaluation scripts that consume chunk bitstreams.
