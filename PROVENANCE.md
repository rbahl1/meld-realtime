# Provenance

This file lists every external file the system uses: where it comes from, its sha256, its licence as stated at the source, and what it contributes. The hashes of the model files come from [results/provenance_hashes.json](results/provenance_hashes.json), which `tools/fetch_models.py` writes after it checks each download against the hashes in its own table. The hashes of the MELD CSV files come from [results/data_stats.md](results/data_stats.md). The size, hash and licence field of MELD.Raw come from [results/meld_raw_source.json](results/meld_raw_source.json), which `tools/meld_raw_source.py` writes from the Hugging Face API and the resolve headers of the named revision. The licence statements were read on 2026-10-12 at the places named, and those of the face models and the reply generator on 2026-10-13. Where no statement was found, this file says so.

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

## YuNet face detector (OpenCV zoo)

- URL: `https://huggingface.co/opencv/face_detection_yunet/resolve/3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/face_detection_yunet_2023mar.onnx` (revision `3cc26e7f1014a5ee5d74a42acee58bafc9d0a310`).
- File: `models/vision/face_detection_yunet_2023mar.onnx`, 232589 bytes, sha256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
- Licence: the README of the repository at that revision says that all its files are under the MIT License, and its LICENSE file is the MIT License, "Copyright (c) 2020 Shiqi Yu".
- Contributes: a box, five landmarks and a detection score for each face in a frame, for the MELD video features and for `--vision`.

## FER MobileFaceNet (OpenCV zoo, Progressive Teacher)

- URL: `https://huggingface.co/opencv/facial_expression_recognition/resolve/b473270576eabddaacffd08adb4bed27b3fc54e6/facial_expression_recognition_mobilefacenet_2022july.onnx` (revision `b473270576eabddaacffd08adb4bed27b3fc54e6`).
- File: `models/vision/facial_expression_recognition_mobilefacenet_2022july.onnx`, 4791892 bytes, sha256 `4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9`.
- Licence: the README says that all files in its directory are under the Apache 2.0 License and links a LICENSE file. The repository has no LICENSE file at that revision.
- Training data: the README names no training set. It reports an accuracy of 88.27 % on RAF-DB and links the RAF-DB page (`http://whdeng.cn/RAF/model1.html`). The method's paper (Jiang and Deng, "Boosting Facial Expression Recognition by A Semi-Supervised Progressive Teacher", arXiv 2205.14361) runs its experiments on RAF-DB and FERPlus. The weights are therefore taken to be trained on RAF-DB. The dataset catalog hyper.ai lists the RAF-DB licence as "Non-Commercial". The RAF-DB page itself could not be reached from the build environment, so its terms were not read at the source. The Apache 2.0 statement covers the files of the model. It does not lift the terms of the data the weights were learned from. This repository uses the model for research only, and any other use needs the RAF-DB terms checked first.
- Contributes: the expression logits of the seven classes and the embedding of each face, from which the vision calibrator gives the face posterior.

## Qwen2.5-0.5B-Instruct (ONNX export by onnx-community)

- URL: `https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/resolve/22942cb7d7ba4cc81bb4673549ca4d4614469b5e/<file>` (revision `22942cb7d7ba4cc81bb4673549ca4d4614469b5e`).

| kept as (under `models/llm/Qwen2.5-0.5B-Instruct/`) | bytes | sha256 | use |
|---|---|---|---|
| `onnx/model_int8.onnx` | 512096557 | `41834041ab1b29eff9fc592f1a29a1844133aea35832ea9fa91682be13016100` | the deployed reply generator |
| `onnx/model_q4.onnx` | 786156820 | `09235a3b1c135cd04ef570e5053b5c079e028078a2cc5f76ba6a251e91bf3296` | generator bench only |
| `tokenizer.json` | 7031673 | `a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf` | tokenizer |
| `config.json` | 678 | `777e01f0fbb3346eb229cb6fb278ed6533c1e4dcb9ebf4bed0f6e94ef17fa1b5` | model settings |
| `generation_config.json` | 242 | `f7e7ce458658b2d40d9eb213b91b77a8bf698845ab89360976722d7ac46928a3` | stop tokens |

- Licence: the onnx-community model card names no licence, and its `base_model` is `Qwen/Qwen2.5-0.5B-Instruct`. The model card of `Qwen/Qwen2.5-0.5B-Instruct` (revision `7ae557604adf67be50417f59c2c2f167def9a775`) gives the licence `apache-2.0`, and its LICENSE file is the Apache License, Version 2.0.
- Contributes: the reply streamed with `--reply llm`. `--groups llm` of `tools/fetch_models.py` fetches the deployed file with the three side files, and `--groups all` also the q4 file.

## Qwen2.5-1.5B-Instruct (ONNX export by onnx-community, not deployed)

- URL: `https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct/resolve/d54957e3af1be04546436e874a60fe386041b244/<file>` (revision `d54957e3af1be04546436e874a60fe386041b244`).

| kept as (under `models/llm/Qwen2.5-1.5B-Instruct/`) | bytes | sha256 |
|---|---|---|
| `onnx/model_int8.onnx` | 1578954293 | `ee8364373c6fb35644c67fd8127cbee6c3d98ac889f8bb32ea4ac04a29650787` |
| `onnx/model_q4.onnx` | 1787566590 | `70c24509f760fed9a8f50391165c46a87660197028a42c4547763f1c173ddea6` |
| `tokenizer.json` | 7031673 | `a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf` |
| `config.json` | 809 | `215eb99c4955b0c42ea9f6e0980d922c228950c5dbc09bde6dc451fbba4d21f3` |
| `generation_config.json` | 242 | `f7e7ce458658b2d40d9eb213b91b77a8bf698845ab89360976722d7ac46928a3` |

- Licence: the onnx-community model card names no licence, and its `base_model` is `Qwen/Qwen2.5-1.5B-Instruct`. The model card of `Qwen/Qwen2.5-1.5B-Instruct` (revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`) gives the licence `apache-2.0`, and its LICENSE file is the Apache License, Version 2.0.
- Contributes: only the generator bench ([results/generator_bench_vm_1.5b.json](results/generator_bench_vm_1.5b.json)). Its files miss the decode-rate target of the choice rule, so the model is not deployed. `--groups all` fetches them.

## MELD annotation CSVs

| file | URL | bytes | sha256 |
|---|---|---|---|
| `meld/train_sent_emo.csv` | `https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv` | 1119290 | `d2fa2d6529cf03cac2989efec05c9b27d8fd2f4c8fc5974c7ae88aa537fa02db` |
| `meld/dev_sent_emo.csv` | `https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv` | 121640 | `2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c` |
| `meld/test_sent_emo.csv` | `https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv` | 294526 | `8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888` |

- Licence: the LICENSE file of the repository (github.com/declare-lab/MELD) is the GNU General Public License, version 3. The README asks users to cite the MELD papers. The utterances are dialogue from the TV series Friends, so the CSV files are fetched by `tools/fetch_meld_csv.py` and not committed. The repository keeps statistics derived from them ([results/data_stats.md](results/data_stats.md)). The two generator bench files also quote CSV text: the prompts of the 100 gold-text states of each run hold their utterances ([results/generator_bench_vm.json](results/generator_bench_vm.json), [results/generator_bench_vm_1.5b.json](results/generator_bench_vm_1.5b.json)).
- Contributes: the splits, the emotion and sentiment labels, and the gold text for training and evaluation.

## MELD.Raw (audio and video)

- Source: the Hugging Face dataset `declare-lab/MELD` at revision `9abc51ee7903424ffb971297608aa6d3d0de3bfa`, file `MELD.Raw.tar.gz`, URL `https://huggingface.co/datasets/declare-lab/MELD/resolve/9abc51ee7903424ffb971297608aa6d3d0de3bfa/MELD.Raw.tar.gz`.
- File: 10878146150 bytes, LFS sha256 `a56b4407d574195cbce470d86f9c9d72fcfea59b0e34502ecd4babee4a5c613e` (`x-linked-size` and `x-linked-etag` in [results/meld_raw_source.json](results/meld_raw_source.json)).
- Download: the build container fetched `resolve/main`, which serves revision `9abc51ee7903424ffb971297608aa6d3d0de3bfa` (`x-repo-commit` in [results/meld_raw_source.json](results/meld_raw_source.json)), and checked the size only. The sha256 of the download was not computed before the archive was deleted.
- Licence: the licence field of the dataset card is `gpl-3.0` (`card_license` in [results/meld_raw_source.json](results/meld_raw_source.json)), which matches the LICENSE file of the GitHub repository above. The clips are cut from Friends, like the CSV text.
- Use: the archive was downloaded into the build container, `tools/stream_meld_raw.py` streamed its clips through `tools/extract_audio.py` into 16 kHz mono WAV files, `tools/cache_audio_features.py` cached the SenseVoice and prosody features, and the archive was deleted. No audio, video or WAV file is in the repository. `audio/` and `cache/` are not committed.
- Video use: a second download of `MELD.Raw.tar.gz`, of the same size, was read as a stream by `tools/extract_video_features.py`. It decoded the frames of each clip at 10 frames per second (the rate recorded for train and dev in `vision.video_cache` of [heads/manifest.json](heads/manifest.json)) and ran YuNet and the FER model on them. It kept only face features per frame (boxes, landmarks, scores, track ids, expression logits, mouth and face motion, embeddings) in `cache/video_{split}.npz`, which is not committed. No sha256 of that download is recorded, and the archive was not kept. The MP4 files of the test split were kept in a folder outside the repository for the three-input demo and the replay bench with video. No frame, face crop or video file is in the repository. The replay bench files and the three-input demo log hold the SenseVoice transcripts of the clips they streamed.
- Missing clips: one train row (`125_3`, an MP4 that ffmpeg could not open) and one dev row (`110_7`, no clip found in the archive) have no audio, and the audio stage leaves them out (`audio_rows_dropped` and `audio_fusion.rows` in [heads/manifest.json](heads/manifest.json)). Every audio row in [results/metrics.json](results/metrics.json) scores all 2610 test utterances.
- Contributes: the audio for the audio calibrators, the fusion head, the fused sentiment head, the dev JSD threshold, every audio and fused test row, the MELD dev pause statistics and the MELD replay bench. The video gives the face features of the vision calibrator, fusion3, the three-input sentiment head and the commit traces, every three-input test row, the three-input demo and the replay bench with video.
- The MELD README also links the archive at `http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz`. That copy was not used, and the two were not compared.

## Papers

The published comparison rows come from three PDFs kept in `papers/` (not committed): the MELD paper (Poria et al., ACL 2019, arXiv 1810.02508), EmoBERTa (Kim and Vossen, arXiv 2108.12009) and the FunAudioLLM report (arXiv 2407.04051). `tools/evaluate.py` holds the rows with their table numbers.

## Made here

These are generated by the tools in this repository, not downloaded.

| what | made by | committed |
|---|---|---|
| `heads/*.npz`, `heads/manifest.json` (sha256 of each head in the manifest) | `tools/train_heads.py` | yes |
| `heads/vision_cal.npz`, `heads/fusion3.npz`, `heads/sentiment3.npz`, section `vision` of `heads/manifest.json` | `tools/train_vision.py` | yes |
| `heads/commit_policy.npz` | `tools/train_commit_policy.py` | yes |
| `models/sensevoice/model_enc.int8.onnx` | `tools/edit_sensevoice_graph.py` | no |
| `audio/{split}/*.wav` (16 kHz MELD clips) | `tools/extract_audio.py` | no |
| `cache/text_{split}.npy`, `cache/text_asr_test*.npy` | `tools/embed_text.py` | no |
| `cache/audio_{split}.npz` | `tools/cache_audio_features.py` | no |
| `cache/video_{split}.npz` (face features of the MELD clips) | `tools/extract_video_features.py` | no |
| `cache/envelope_{split}.npz`, `cache/folds/`, `cache/vision_post_{split}.npz` | `tools/train_vision.py` | no |
| `cache/commit_traces_{split}.npz` | `tools/build_commit_traces.py` | no |
| `proxy/` (100 Piper clips of MELD dev text) | `tools/make_proxy_clips.py` | no |
| `bench_clips/` (cuts and tilings of `en.wav`) | `tools/bench_components.py` | no |
| `meld_rt/response/bank.yaml` (124 reply lines) | written for this repository | yes |
| `meld_rt/response/lexicon.py` (the emotion words of each class, which the reply generator masks) | written for this repository | yes |
