# CTC Decoder Modules

This repository provides two implementations of **CTC (Connectionist Temporal Classification)** beam search decoders for speech recognition models, built on top of [PyTorch](https://pytorch.org/) and [torchaudio](https://pytorch.org/audio/stable/index.html).  
The decoders convert model emission probabilities into human-readable transcriptions.

---

## 📦 Files

### 1. `_ctc_decoder.py`
**CPU-based Flashlight CTC Decoder**

- Implements a CTC beam search decoder using the [Flashlight text decoder](https://github.com/flashlight/text).
- Supports **lexicons**, **KenLM language models**, and **word-level beam search**.
- Designed for high-accuracy decoding (e.g., LibriSpeech, large vocabulary ASR systems).

**Key features**
- Lexicon-based and lexicon-free decoding
- KenLM and custom language model support
- Configurable beam size, beam threshold, and scoring parameters
- Returns N-best hypotheses with token and word alignment

**Main Classes**
- `CTCDecoder`: Main beam search decoder
- `CTCDecoderLM`: Language model interface
- `CTCHypothesis`: Decoding result structure
- Factory function: `ctc_decoder(...)`

---

### 2. `_cuda_ctc_decoder.py`
**GPU-accelerated CTC Decoder**

- A fast CUDA implementation of prefix beam search decoding.
- Uses `torchaudio.lib.pybind11_prefixctc` (`libctc_prefix_decoder`) for GPU inference.
- Ideal for real-time or streaming ASR applications.

**Key features**
- Fully GPU-based beam search decoding
- Very fast batched decoding on CUDA
- Supports skipping frames with high blank probabilities
- Returns N-best hypotheses directly from GPU tensors

**Main Classes**
- `CUCTCDecoder`: GPU-based beam search decoder
- `CUCTCHypothesis`: Result structure
- Factory function: `cuda_ctc_decoder(...)`

---

## ⚙️ Installation Requirements

Both decoders require:
```bash
pip install torch torchaudio flashlight-text
