<p align="center"><img src="media/miner-llama.jpg" alt="Miner Llama: a llama in a miner's helmet over two crossed pickaxes" width="240"></p>

<h1 align="center">Miner Llama</h1>
<p align="center"><b>llama.cpp for retired GPU-mining rigs and other Pascal cards (compute 6.1)</b></p>

<p align="center">
  <a href="https://github.com/Cinderella-Man/miner-llama/releases/latest"><img alt="release" src="https://img.shields.io/github/v/release/Cinderella-Man/miner-llama?color=d4a017"></a>
  <a href="https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs"><img alt="docs" src="https://img.shields.io/badge/docs-llms--on--crypto--mining--rigs-blue"></a>
  <a href="https://huggingface.co/CinderellaMan"><img alt="models" src="https://img.shields.io/badge/models-Hugging%20Face-ffcc4d"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

Miner Llama is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for NVIDIA's Pascal cards, built and
measured on a Manli mining rig: nine P106-090 cards with 6 GB each, every card on a single PCIe 1.0 lane, and a
2-core Celeron host with 4 GB of RAM. Its kernels, its multi-card pipeline and its model files make that rig run
today's models for a coding agent:

| A coding agent on nine P106 cards, two runs each | Miner Llama v3 | **Miner Llama v4** |
|---|---:|---:|
| **Qwen3.8-Flash-Next 125B, 2-bit:** tokens written per second | 20.66, 18.92 | **27.61, 26.67** |
| minutes a task, at equal work | 3.08, 3.19 | **2.37, 2.43** |
| 49,152 tokens into a conversation: prompt tok/s | 70.3, 70.6 | **180.7, 180.8** |
| 49,152 tokens into a conversation: tokens written per second | 10.66, 10.98 | **24.26, 24.14** |
| **Ornith-1.5-122B, 2–3-bit:** minutes a task, at equal work | 2.52, 2.54 | **2.11, 2.13** |
| prompt tok/s | 61.1, 58.4 | **103.0, 105.9** |
| **Flint 39.8B, 8-bit:** seconds to the first token of a turn | 2.65, 2.61 | **1.74, 1.82** |
| tokens written per second | 48.55, 48.02 | 49.32, 46.99 |

Every value is one run on the same rigs, each engine with the model's own preset; "equal work" is the same prompt and
reply tokens at each run's own speeds. The files are UkisAI's Swift-1.5 GSQ-RCO Q2_0 and my own
[Ornith-1.5-122B Pascal N2P GGUF](https://huggingface.co/CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF) and
[Flint Pascal Q8 GGUF](https://huggingface.co/CinderellaMan/Flint-Pascal-Q8-GGUF). Flint and Ornith write one token a
step, so v4 changes how fast they read, not how fast they write.

Against stock llama.cpp, measured for v3 (stock was not measured again for v4):

| A coding agent, tok/s | Stock llama.cpp b12076 | **Miner Llama v3** | **Miner Llama v3, my quant** |
|---|---:|---:|---:|
| Qwen3.8-Flash-Next 125B, 2-bit | 8.20 | **20.40** (2.48×) | — |
| Flint 39.8B, 8-bit | 14.4 | **45.9** (3.18×) | **47.4**<br>[Flint Pascal Q8 GGUF](https://huggingface.co/CinderellaMan/Flint-Pascal-Q8-GGUF) |
| Flint 39.8B, 5-bit | 13.7 | **46.4** (3.38×) | **49.4**<br>[Flint Pascal Q5 GGUF](https://huggingface.co/CinderellaMan/Flint-Pascal-Q5-GGUF) |
| Ornith-1.5-122B, 2-bit | 6.0 | **12.6** (2.10×) | **16.3**<br>[Ornith-1.5-122B Pascal N2P GGUF](https://huggingface.co/CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF) |

The public files are UkisAI's Swift-1.5 GSQ-RCO Q2_0 and Unsloth's Q8_0, UD-Q5_K_XL and UD-Q2_K_XL. Stock llama.cpp
runs Flint at 65,536 tokens of context, because at 262,144 its attention mask does not fit in the 4 GB host; Miner Llama
runs the full 262,144. Stock reached 9 of Flash-Next's 12 tasks in its time budget; on those nine Miner Llama v3
generated 20.25 tok/s. Flint Pascal Q5 GGUF's 49.4 includes one long session (complex-numbers, 14,024 tokens); on the other
eleven tasks it generated 53.64 tok/s, as on v2.0 (54.04). Ornith's 16.3 ran on `1f7215948`, the release branch before
its last fix; on the release build `1551f5fe1` the same preset ran 16.42 on a second rig.

Every number, with its command and data, is in the docs repository,
[llms-on-crypto-mining-rigs](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs).

## Get it

The [v4.0 release](https://github.com/Cinderella-Man/miner-llama/releases/tag/miner-llama-v4.0) has the binaries for
Linux (Ubuntu 24.04, NVIDIA's driver 580) and the CUDA runtime, so no CUDA toolkit is needed. How to set up the rig
and run each model with its preset:
- [Qwen3.8-Flash-Next](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flash-next-v4/README.md)
- [Flint](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v4/README.md)
- [Ornith-1.5-122B](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/ornith-v4/README.md)

What each switch does: [docs/switches.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/switches.md). The release notes: [notes-v4.0.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/notes-v4.0.md)

## Branches and tags

- `miner-llama` (this branch, the default): the latest release's code, `eb82916d3` for v4.0, plus this page.
- `miner-llama-v4.0`, `miner-llama-v3.0`, `miner-llama-v2.0`: each release's exact code, as tags.
- `pascal-p106-v4`: the v4.0 engine, every feature behind its switch. `pascal-p106-v3-b12076-ladder5` is v3.0's
  engine, `pascal-p106-v2` v2.0's, `pascal-p106` v1's.
- `pascal-p106-v5`: where the work after v4.0 collects.
- `master`: upstream llama.cpp.

## Upstream

Everything not listed above is llama.cpp's own, and its README is kept as [README-llama.cpp.md](README-llama.cpp.md).
Miner Llama is MIT-licensed like llama.cpp ([LICENSE](LICENSE)) and is not affiliated with ggml-org. The fixes that
belong upstream are written up in [docs/upstream_fixes.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/upstream_fixes.md).
