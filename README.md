# japanese-rag-kotlin

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23122937.svg)](https://doi.org/10.5281/zenodo.23122937)

[English](#english) | [日本語](#日本語)

## English

**On-device Japanese RAG for Android / JVM in Kotlin.** A pure-Kotlin core (a ruri-v3 **Unigram tokenizer**, Japanese sentence chunker,
in-memory vector index, retrieve -> rerank pipeline) plus an Android module that runs the **LiteRT (`.tflite`) conversions of ruri-v3 and the
Japanese rerankers**: [ruri-v3-30m/130m/310m](https://huggingface.co/masahiroid/ruri-v3-130m-tflite),
[japanese-reranker-xsmall/small-v2](https://huggingface.co/masahiroid/japanese-reranker-small-v2-tflite),
[ruri-v3-reranker-310m](https://huggingface.co/masahiroid/ruri-v3-reranker-310m-tflite). No LLM is included: use it to choose the passages you hand to an
on-device LLM. Counterpart of [japanese-rag-swift](https://github.com/masahiroid/japanese-rag-swift).

| Module | Contents |
|---|---|
| `:core` (JVM, no Android dependency) | `UnigramTokenizer` (reads the HF `tokenizer.json`), `JapaneseChunker`, `VectorIndex`, `RagPipeline`, `RuriPromptPrefix` / `PrefixedEmbedder` |
| `:litert` (Android library) | `LiteRtEmbedder`, `LiteRtReranker` on `com.google.ai.edge.litert:litert` (int32 `input_ids` + `attention_mask`, fixed sequence length) |

```kotlin
val tokenizer = UnigramTokenizer.fromJson(File("tokenizers/ruri-v3/tokenizer.json").readText())   // or load from assets
val model = mapModel(File(filesDir, "ruri-v3-130m_seq128.tflite"))
val base = LiteRtEmbedder(model, tokenizer, sequenceLength = 128, dimension = 512)                 // 30m: 256, 130m: 512, 310m: 768

val rag = RagPipeline(
    queryEmbedder = PrefixedEmbedder(base, RuriPromptPrefix.SEARCH_QUERY),
    documentEmbedder = PrefixedEmbedder(base, RuriPromptPrefix.SEARCH_DOCUMENT),
)
rag.ingest("tower", "東京タワーは港区にある電波塔で、高さは333メートルです。")
val hits = rag.search("港区にある電波塔", retrieveK = 20, topN = 5)
```

**Why a Kotlin tokenizer**: the `.tflite` graphs take token ids, so an Android app needs the exact tokenizer. `UnigramTokenizer` implements the ruri-v3 pipeline
(Metaspace, Unigram Viterbi, byte fallback, `<s> ... </s>` template) and **reproduces the Hugging Face ids exactly on 450/450 bundled fixtures** (Japanese prompts, injection texts,
half/full-width kana, astral kanji, emoji, control characters, pairs for rerankers; `core/src/test`).

**Verified**: **on-device run on the Android emulator** (API 36.1, arm64) with the `:sample` app: Kotlin tokenizer -> `ruri-v3-30m_seq128.tflite` -> retrieval ranked 3/3 sample queries correctly (e.g. 「日本で一番高い山は？」 -> the Mt. Fuji passage), and `japanese-reranker-xsmall-v2_seq256.tflite` ranked the Tokyo Tower passage first (~100-140 ms per query on the emulator CPU; **not** a real-device figure). `gradle :core:test` (tokenizer fixtures + chunker, index, pipeline tests) passes; `gradle :litert:assembleDebug` builds the AAR
(AGP 9.4.1, LiteRT 2.2.0, compileSdk 36). **Not yet verified**: latency and memory on real devices, GPU/NNAPI delegates, a richer sample UI. The fp32 models are large (30m ~150 MB, 130m ~530 MB, 310m ~1.3 GB); quantized variants are future work.

**Run the sample**: `gradle :sample:assembleDebug`, install the APK, then `adb push ruri-v3-30m_seq128.tflite` (and optionally `japanese-reranker-xsmall-v2_seq256.tflite`) to `/sdcard/Android/data/jp.masahirocom.ragkit.sample/files/`, launch the app and read logcat tag `RAGSAMPLE`. (`gradle.properties` sets `android.uniquePackageNames=false` because LiteRT's `litert` and `litert-api` AARs share a namespace, which AGP 9 otherwise rejects.)

Requirements: JDK 17+, Gradle 9, Android SDK (for `:litert`; set `sdk.dir` in `local.properties`). License: Apache-2.0 (matching ruri-v3).

## 日本語

**Kotlinで書いた、Android / JVM向けのオンデバイス日本語RAG**です。純Kotlinのcore（ruri-v3の**Unigramトークナイザー**、日本語の文チャンク分割、
メモリ上のベクトル索引、検索→再ランキングのパイプライン）と、ruri-v3と日本語リランカーの**LiteRT（`.tflite`）変換**を動かすAndroidモジュールで構成します。
LLMは含みません。端末上のLLMに渡す文章を選ぶ部分として使います。[japanese-rag-swift](https://github.com/masahiroid/japanese-rag-swift) のKotlin版です。

使い方は上の English セクションのコードを参照してください（モデルは各 `masahiroid/*-tflite` リポジトリ）。

**Kotlinのトークナイザーが必要な理由**: `.tflite` はトークンIDを入力に取るため、Androidアプリには正確なトークナイザーが要ります。`UnigramTokenizer` は
ruri-v3のパイプライン（Metaspace、UnigramのViterbi、バイトフォールバック、`<s> ... </s>`）を実装し、同梱の450件のフィクスチャ（日本語の指示文・注入文、
半角・全角カナ、補助平面の漢字、絵文字、制御文字、リランカー用のペア）で、**Hugging Faceと同一のIDを完全に再現**します（`core/src/test`）。

**検証済み**: **Androidエミュレータ（API 36.1、arm64）での端末上実行**。`:sample` アプリで、Kotlinのトークナイザー → `ruri-v3-30m_seq128.tflite` → 検索が3つのサンプルクエリを全て正しく順位付け（例: 「日本で一番高い山は？」→ 富士山の文章が1位）、`japanese-reranker-xsmall-v2_seq256.tflite` が東京タワーの文章を1位にしました（エミュレータのCPUで1クエリ約100〜140ms。**実機の値ではありません**）。`gradle :core:test`（トークナイザー、チャンク、索引、パイプライン）が通り、`gradle :litert:assembleDebug` でAARがビルドできます
（AGP 9.4.1、LiteRT 2.2.0、compileSdk 36）。**未検証**: 実機のレイテンシとメモリ、GPU／NNAPIデリゲート、リッチなサンプルUI。fp32モデルは大きいです（30m 約150MB、130m 約530MB、310m 約1.3GB）。量子化版は今後の課題です。

要件: JDK 17+、Gradle 9、Android SDK（`:litert` 用。`local.properties` に `sdk.dir`）。ライセンス: Apache-2.0（ruri-v3に合わせています）。
