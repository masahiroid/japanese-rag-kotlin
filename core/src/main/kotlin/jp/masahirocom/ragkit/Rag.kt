package jp.masahirocom.ragkit

import kotlin.math.exp
import kotlin.math.sqrt

/** Turns text into a (usually L2-normalized) embedding. */
fun interface SentenceEmbedder { suspend fun embed(text: String): FloatArray }

/** Scores how well a document answers a query (higher is better). */
fun interface Reranker { suspend fun score(query: String, document: String): Float }

/** ruri-v3 prefix scheme ("1+3"): the same encoder works better when the text carries its role. */
enum class RuriPromptPrefix(val value: String) {
    NONE(""), TOPIC("トピック: "), SEARCH_QUERY("検索クエリ: "), SEARCH_DOCUMENT("検索文書: ");

    fun applied(text: String) = value + text
}

/** Applies a fixed prefix before delegating, so one model can serve as query and document embedder. */
class PrefixedEmbedder(private val base: SentenceEmbedder, private val prefix: RuriPromptPrefix) : SentenceEmbedder {
    override suspend fun embed(text: String) = base.embed(prefix.applied(text))
}

object VectorMath {
    fun dot(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Vectors must have the same dimensionality (${a.size} vs ${b.size})." }
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }

    fun l2Normalized(v: FloatArray): FloatArray {
        var sum = 0f
        for (x in v) sum += x * x
        val m = sqrt(sum)
        return if (m > Float.MIN_VALUE * 1e6f) FloatArray(v.size) { v[it] / m } else v
    }

    fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()
}

/**
 * Splits Japanese text for embedding: sentence boundaries (。！？!? and newlines) are kept, sentences are packed up to
 * [maxCharacters], and [overlapSentences] trailing sentences are repeated at the start of the next chunk.
 */
class JapaneseChunker(maxCharacters: Int = 200, overlapSentences: Int = 1) {
    val maxCharacters = maxOf(maxCharacters, 1)
    val overlapSentences = maxOf(overlapSentences, 0)

    fun sentences(text: String): List<String> {
        val out = mutableListOf<String>()
        val cur = StringBuilder()
        for (ch in text) {
            cur.append(ch)
            if (ch in TERMINATORS) {
                val s = cur.toString().trim()
                if (s.isNotEmpty()) out += s
                cur.setLength(0)
            }
        }
        val tail = cur.toString().trim()
        if (tail.isNotEmpty()) out += tail
        return out
    }

    fun chunk(text: String): List<String> {
        val units = mutableListOf<String>()
        for (s in sentences(text)) {
            var rest = s
            while (rest.length > maxCharacters) { units += rest.take(maxCharacters); rest = rest.drop(maxCharacters) }
            if (rest.isNotEmpty()) units += rest
        }
        val chunks = mutableListOf<String>()
        var window = mutableListOf<String>()
        var length = 0
        for (u in units) {
            if (length + u.length > maxCharacters && window.isNotEmpty()) {
                chunks += window.joinToString("")
                val keep = if (overlapSentences > 0) window.takeLast(overlapSentences) else emptyList()
                window = keep.filter { it.length + u.length <= maxCharacters }.toMutableList()
                length = window.sumOf { it.length }
            }
            window += u
            length += u.length
        }
        if (window.isNotEmpty()) chunks += window.joinToString("")
        return chunks
    }

    private companion object { val TERMINATORS = setOf('。', '！', '？', '!', '?', '\n') }
}

data class IndexedChunk(val id: String, val text: String, val vector: FloatArray, val metadata: Map<String, String> = emptyMap()) {
    override fun equals(other: Any?) = other is IndexedChunk && id == other.id && text == other.text &&
        vector.contentEquals(other.vector) && metadata == other.metadata
    override fun hashCode() = id.hashCode() * 31 + text.hashCode()
}

data class SearchHit(val chunk: IndexedChunk, val score: Float)
data class RankedPassage(val chunk: IndexedChunk, val retrievalScore: Float, val rerankScore: Float?)

/** Exact in-memory search by dot product (vectors expected L2-normalized); replace-by-id. */
class VectorIndex {
    private val items = LinkedHashMap<String, IndexedChunk>()
    var dimension: Int? = null
        private set
    val size get() = items.size
    val chunks: List<IndexedChunk> get() = items.values.toList()

    fun add(chunk: IndexedChunk) {
        dimension?.let { require(it == chunk.vector.size) { "Vector dimension ${chunk.vector.size} does not match the index dimension $it." } }
        dimension = chunk.vector.size
        items[chunk.id] = chunk
    }

    fun remove(id: String) { items.remove(id) }

    fun search(query: FloatArray, topK: Int): List<SearchHit> {
        dimension?.let { require(it == query.size) { "Query dimension ${query.size} does not match the index dimension $it." } }
        return items.values.map { SearchHit(it, VectorMath.dot(query, it.vector)) }.sortedByDescending { it.score }.take(maxOf(topK, 0))
    }
}

/** ingest text -> chunk -> embed (document prefix) -> index; query -> embed (query prefix) -> top-K -> rerank -> top-N. */
class RagPipeline(
    private val queryEmbedder: SentenceEmbedder,
    private val documentEmbedder: SentenceEmbedder,
    private val reranker: Reranker? = null,
    private val chunker: JapaneseChunker = JapaneseChunker(),
    val index: VectorIndex = VectorIndex(),
) {
    suspend fun ingest(documentId: String, text: String, metadata: Map<String, String> = emptyMap()): Int {
        val pieces = chunker.chunk(text)
        pieces.forEachIndexed { i, p ->
            index.add(IndexedChunk("$documentId#$i", p, VectorMath.l2Normalized(documentEmbedder.embed(p)), metadata))
        }
        return pieces.size
    }

    suspend fun search(query: String, retrieveK: Int = 20, topN: Int = 5): List<RankedPassage> {
        val hits = index.search(VectorMath.l2Normalized(queryEmbedder.embed(query)), retrieveK)
        if (reranker == null) return hits.take(topN).map { RankedPassage(it.chunk, it.score, null) }
        return hits.map { RankedPassage(it.chunk, it.score, reranker.score(query, it.chunk.text)) }
            .sortedByDescending { it.rerankScore ?: 0f }.take(topN)
    }
}
