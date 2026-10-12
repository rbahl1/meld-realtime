# Provenance

This file lists every external file the system uses: where it comes from, its sha256, its licence as stated at the source, and what it contributes. The hashes of the model files come from [results/provenance_hashes.json](results/provenance_hashes.json), which `tools/fetch_models.py` writes after it checks each download against the hashes in its own table. The hashes of the MELD CSV files come from [results/data_stats.md](results/data_stats.md). The licence statements were read on 2026-10-12 at the places named. Where no statement was found, this file says so.

## SenseVoice-small (sherpa-onnx release of 2024-07-17)

- URL: `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2`. The archive is downloaded to a temporary folder and deleted, so its own hash is not recorded. `tools/fetch_models.py` keeps and checks these members:

| kept as | bytes | sha256 |
|---|---|---|
| `models/sensevoice/model.int8.onnx` | 239233841 | `c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51` |
| `models/sensevoice/tokens.txt` | 315894 | `f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc` |
| `models/sensevoice/test_wavs/` | 5 files | not hashed, existence only |

- Licence: the README of the SenseVoice repository (github.com/FunAudioLLM/SenseVoice, section "License") says that the source code is under the MIT License and that "Model weights are distributed separately and follow the terms on each model card." It says that the official SenseVoiceSmall model card links to the FunASR Model Open Source License Agreement (github.com/modelscope/FunASR, file `MODEL_LICENSE`, version 1.1 when read). It also says that "other artifacts and conversions may list different terms, so check their model cards before use." The int8 ONNX file used here is such a conversion, published by the sherpa-onnx project. No licence statement specific to this conversion was found: the archive members other than the ones kept were not inspected, and the Hugging Face model card could not be read from the build environment. The licence of the conversion is therefore unknown beyond the upstream statement.
- Contributes: the transcript, the ASR confidence, the language, the 9-way emotion posterior, the 11-way event posterior and, through the edited graph, the encoder states. `en.wav` from `test_wavs/` is the demo clip and the source of the bench clips.

The FunAudioLLM report describes how SenseVoice got its emotion labels (arXiv 2407.04051, Section 3.1, footnote markers left out):

> The SenseVoice-Small model was trained on an extensive audio data corpus of approximately 300,000 hours, covering 5 languages including Chinese, Cantonese, English, Japanese, and Korean. [...] To obtain rich transcription labels from speech data, we leveraged open-source models for audio event detection (AED) and speech emotion recognition (SER) to generate pseudo labels, yielding an extensive rich transcribe dataset. Specifically, the AED data amounted to 150 million entries, while the SER data comprised 30 million entries.

The footnote names the SER model only by a link (`https://modelscope.cn/models/iic/emotion2vec_plus_large`). The report does not list the audio that was labelled or the data behind that SER model. Its Section 4.2 evaluates on MELD among other sets and says that "SenseVoice achieves a good performance on all test sets and all metrics even without fine-tuning on the target domain", with a MELD weighted F1 of 57.7 for SenseVoice-S (Table 8).

The consequence: MELD audio, or MELD labels through the SER model, may be in SenseVoice's training data, and nothing here can rule that out. The audio-only number of this repository may be described as SenseVoice-small used without fine-tuning on MELD. It may not be described as zero-shot, or as a result on unseen data. The heads trained here use MELD train and dev only, and test is read once per condition, but that does not remove an overlap upstream.

## Silero VAD v4

- URL: `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx`
- File: `models/silero_vad.onnx`, 643854 bytes, sha256 `9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6`.
- Licence: the file is a bare ONNX graph with no licence text. The LICENSE file of the upstream repository (github.com/snakers4/silero-vad) is the MIT License, "Copyright (c) 2020-present Silero Team". No separate statement from the sherpa-onnx release was checked.
- Contributes: the speech probability of every 32 ms frame, from which the service finds the start and end of each utterance.

## RoBERTa-base (ONNX model zoo)

- URL: `https://media.githubusercontent.com/media/onnx/models/main/validated/text/machine_comprehension/roberta/model/roberta-base-11.onnx`
- File: `models/roberta-base-11.onnx`, 498649858 bytes, sha256 `ad476a33a4b227f6e6b2e1c7192df1b61640657f26b390857ab943de66236c0b`.
- Licence: the model's README in the zoo (github.com/onnx/models, `validated/text/machine_comprehension/roberta/README.md`) ends with the section "License" and the text "Apache 2.0". The original roberta-base weights come from fairseq, whose repository LICENSE file (github.com/facebookresearch/fairseq) is the MIT License.
- Contributes: the frozen text encoder, 768 values per utterance.

## spaCy en_core_web_trf 3.8.0 wheel and the tokenizer files

- URL: `https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl`
- File: `verify_dl/en_core_web_trf-3.8.0-py3-none-any.whl`, 457421864 bytes, sha256 `272a31e9d8530d1e075351d30a462d7e80e31da23574f1b274e200f3fff35bf5`.
- Licence: the wheel's `meta.json` and `METADATA` give "MIT" for the package (Explosion). `meta.json` lists four sources with their own entries: OntoNotes 5 ("commercial (licensed by Explosion)"), the ClearNLP conversion ("Citation provided for reference, no code packaged with model"), WordNet 3.0 ("WordNet 3.0 License") and roberta-base, whose licence field is empty.
- Derived files, committed: `assets/roberta_tokenizer/vocab.json` (50265 pieces) and `assets/roberta_tokenizer/merges.txt` (50000 merges). `tools/extract_tokenizer.py` reads them from the `byte_bpe_processor` attribute inside the wheel and checks the counts and the known ids. Their sha256 is not recorded in `results/`, and git records their content. They are the roberta-base byte-level BPE vocabulary, so they carry the wheel's MIT statement and the empty roberta-base entry above. fairseq, the source named for roberta-base, is under the MIT License.
- Contributes: the tokenizer of the text expert. The wheel's own transformer weights are not used, and the wheel is only needed to rerun the extraction.

## Piper voice en_US-lessac-medium

- URL: `https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2`. The archive is not kept. These members are:

| kept as (under `models/piper/en_US-lessac-medium/`) | bytes | sha256 |
|---|---|---|
| `en_US-lessac-medium.onnx` | 63149198 | `4ba07d8549906668ee855fd9abf9faf66c5db74742712ff026a159f7277fca9f` |
| `en_US-lessac-medium.onnx.json` | 4885 | `efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0` |
| `tokens.txt` | 921 | `87c8ef66eae5473ed0cc0366b3964c736ca6c5f676c979522ea31234e47430b9` |
| `MODEL_CARD` | 351 | `ce49eb457742208166d399a40cdec2c7fa9db77960930031564ab56f12882645` |
| `espeak-ng-data/` | 355 files | not hashed, existence only |

- Licence: the `MODEL_CARD` in the archive names the training data (the lessac voice of the Blizzard Challenge 2013) and links its licence (`https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html`, not read). It gives no licence for the voice weights, so that licence is unknown. The Piper repository (github.com/rhasspy/piper) has an MIT License file for its code. `espeak-ng-data` comes from espeak-ng, whose repository file COPYING is the GNU General Public License, version 3.
- Contributes: the synthetic speech of the proxy clips (`tools/make_proxy_clips.py`). It is not on the inference path.

## MELD annotation CSVs

| file | URL | bytes | sha256 |
|---|---|---|---|
| `meld/train_sent_emo.csv` | `https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv` | 1119290 | `d2fa2d6529cf03cac2989efec05c9b27d8fd2f4c8fc5974c7ae88aa537fa02db` |
| `meld/dev_sent_emo.csv` | `https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv` | 121640 | `2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c` |
| `meld/test_sent_emo.csv` | `https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv` | 294526 | `8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888` |

- Licence: the LICENSE file of the repository (github.com/declare-lab/MELD) is the GNU General Public License, version 3. The README asks users to cite the MELD papers. The utterances are dialogue from the TV series Friends, so the CSV files are fetched by `tools/fetch_meld_csv.py` and not committed, and the repository keeps only statistics derived from them ([results/data_stats.md](results/data_stats.md)).
- Contributes: the splits, the emotion and sentiment labels, and the gold text for training and evaluation.
- MELD.Raw, the audio and video, is not used yet. The MELD README links it at `http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz`.

## Papers

The published comparison rows come from three PDFs kept in `papers/` (not committed): the MELD paper (Poria et al., ACL 2019, arXiv 1810.02508), EmoBERTa (Kim and Vossen, arXiv 2108.12009) and the FunAudioLLM report (arXiv 2407.04051). `tools/evaluate.py` holds the rows with their table numbers.

## Made here

These are generated by the tools in this repository, not downloaded.

| what | made by | committed |
|---|---|---|
| `heads/*.npz`, `heads/manifest.json` (sha256 of each head in the manifest) | `tools/train_heads.py` | yes |
| `models/sensevoice/model_enc.int8.onnx` | `tools/edit_sensevoice_graph.py` | no |
| `cache/text_{split}.npy` | `tools/embed_text.py` | no |
| `cache/audio_{split}.npz` | `tools/cache_audio_features.py` | no, needs MELD audio |
| `proxy/` (100 Piper clips of MELD dev text) | `tools/make_proxy_clips.py` | no |
| `bench_clips/` (cuts and tilings of `en.wav`) | `tools/bench_components.py` | no |
| `meld_rt/response/bank.yaml` (124 reply lines) | written for this repository | yes |
