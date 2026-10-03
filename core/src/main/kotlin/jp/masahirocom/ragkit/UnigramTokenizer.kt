package jp.masahirocom.ragkit

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * A SentencePiece-style Unigram tokenizer that reads a Hugging Face `tokenizer.json` (the ruri-v3 one in `tokenizers/ruri-v3/`).
 * Pipeline: Metaspace (space -> U+2581, no prepended marker, no splitting) -> Viterbi over the vocabulary scores ->
 * byte fallback (`<0xXX>`) for characters outside the vocabulary -> template `<s> A </s>` (pair: `<s> A </s> <s> B </s>`).
 * It reproduces the Hugging Face ids exactly on the bundled fixtures (see TokenizerFixtureTest).
 * Pure Kotlin (JVM / Android), no native code.
 */
class UnigramTokenizer private constructor(
    private val pieces: HashMap<String, Int>,
    private val scores: DoubleArray,
    private val unkId: Int,
    val bosId: Int,
    val eosId: Int,
    val padId: Int,
    private val byteIds: IntArray,
    private val maxPieceLength: Int,
    private val unkScore: Double,
) {
    /** Token ids (without special tokens). */
    fun tokenize(text: String): IntArray {
        val s = text.replace(' ', META)
        val cps = s.codePoints().toArray()
        val n = cps.size
        if (n == 0) return IntArray(0)

        val best = DoubleArray(n + 1) { Double.NEGATIVE_INFINITY }
        val backLen = IntArray(n + 1)
        val backId = IntArray(n + 1)
        best[0] = 0.0
        val sb = StringBuilder()
        for (i in 0 until n) {
            if (best[i] == Double.NEGATIVE_INFINITY) continue
            sb.setLength(0)
            var hasSingle = false
            val limit = minOf(maxPieceLength, n - i)
            for (len in 1..limit) {
                sb.appendCodePoint(cps[i + len - 1])
                val id = pieces[sb.toString()] ?: continue
                if (len == 1) hasSingle = true
                val cand = best[i] + scores[id]
                if (cand > best[i + len]) { best[i + len] = cand; backLen[i + len] = len; backId[i + len] = id }
            }
            if (!hasSingle) {   // unknown character: scored below every real piece, expanded to bytes afterwards
                val cand = best[i] + unkScore
                if (cand > best[i + 1]) { best[i + 1] = cand; backLen[i + 1] = 1; backId[i + 1] = unkId }
            }
        }
        val out = ArrayList<Int>()
        var end = n
        val rev = ArrayList<IntArray>()
        while (end > 0) {
            val len = backLen[end]
            val id = backId[end]
            if (id == unkId) {
                val bytes = String(Character.toChars(cps[end - 1])).toByteArray(Charsets.UTF_8)
                rev.add(IntArray(bytes.size) { k ->
                    val b = bytes[k].toInt() and 0xFF
                    if (byteIds[b] >= 0) byteIds[b] else unkId
                })
            } else rev.add(intArrayOf(id))
            end -= len
        }
        for (k in rev.indices.reversed()) for (id in rev[k]) out.add(id)
        return out.toIntArray()
    }

    /** `<s> text </s>`, truncated to [maxLength] (keeping `</s>`). */
    fun encode(text: String, maxLength: Int = Int.MAX_VALUE): IntArray {
        val body = tokenize(text)
        val room = maxOf(maxLength - 2, 0)
        val cut = if (body.size > room) body.copyOf(room) else body
        return intArrayOf(bosId) + cut + intArrayOf(eosId)
    }

    /** `<s> a </s> <s> b </s>` for cross-encoders; `b` is truncated first. */
    fun encodePair(a: String, b: String, maxLength: Int = Int.MAX_VALUE): IntArray {
        val ta = tokenize(a)
        val tb = tokenize(b)
        val fixed = 4  // <s> </s> <s> </s>
        var keepA = ta.size
        var keepB = tb.size
        if (keepA + keepB + fixed > maxLength) {
            keepB = maxOf(maxLength - fixed - keepA, 0)
            if (keepA + keepB + fixed > maxLength) keepA = maxOf(maxLength - fixed, 0)
        }
        return intArrayOf(bosId) + ta.copyOf(keepA) + intArrayOf(eosId, bosId) + tb.copyOf(keepB) + intArrayOf(eosId)
    }

    /** Pads (with the pad id) or truncates [ids] to [length]; returns ids and the attention mask. */
    fun padToLength(ids: IntArray, length: Int): Pair<IntArray, IntArray> {
        val real = minOf(ids.size, length)
        val padded = IntArray(length) { if (it < real) ids[it] else padId }
        val mask = IntArray(length) { if (it < real) 1 else 0 }
        return padded to mask
    }

    companion object {
        private const val META = '▁'

        fun fromJson(json: String): UnigramTokenizer {
            val root = Json.parseToJsonElement(json).jsonObject
            val model = root["model"]!!.jsonObject
            val vocab: JsonArray = model["vocab"]!!.jsonArray
            val pieces = HashMap<String, Int>(vocab.size * 2)
            val scores = DoubleArray(vocab.size)
            var maxLen = 1
            var minScore = Double.MAX_VALUE
            vocab.forEachIndexed { i, e ->
                val a = e.jsonArray
                val p = a[0].jsonPrimitive.content
                pieces[p] = i
                scores[i] = a[1].jsonPrimitive.double
                maxLen = maxOf(maxLen, p.codePointCount(0, p.length))
                if (i > 5) minScore = minOf(minScore, scores[i])
            }
            val unk = model["unk_id"]!!.jsonPrimitive.int
            val byteIds = IntArray(256) { b -> pieces["<0x%02X>".format(b)] ?: -1 }
            return UnigramTokenizer(
                pieces, scores, unk, pieces["<s>"] ?: 1, pieces["</s>"] ?: 2, pieces["<pad>"] ?: 3,
                byteIds, maxLen, minScore - 10.0,
            )
        }
    }
}
