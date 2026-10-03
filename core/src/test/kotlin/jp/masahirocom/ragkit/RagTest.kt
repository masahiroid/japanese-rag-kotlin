package jp.masahirocom.ragkit

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest

class CharEmbedder : SentenceEmbedder {
    override suspend fun embed(text: String): FloatArray {
        val v = FloatArray(64)
        text.codePoints().forEach { v[it % 64] += 1f }
        return VectorMath.l2Normalized(v)
    }
}

class RagTest {
    @Test fun vectorMath() {
        assertEquals(listOf(0.6f, 0.8f), VectorMath.l2Normalized(floatArrayOf(3f, 4f)).toList())
        assertEquals(listOf(0f, 0f), VectorMath.l2Normalized(floatArrayOf(0f, 0f)).toList())
        assertEquals(0.5f, VectorMath.dot(floatArrayOf(1f, 0f), floatArrayOf(0.5f, 0.5f)))
        assertEquals(0.5f, VectorMath.sigmoid(0f), 1e-6f)
    }

    @Test fun prefixes() {
        assertEquals("検索クエリ: 天気", RuriPromptPrefix.SEARCH_QUERY.applied("天気"))
        assertEquals("検索文書: 晴れ", RuriPromptPrefix.SEARCH_DOCUMENT.applied("晴れ"))
    }

    @Test fun sentenceSplitting() {
        assertEquals(listOf("今日は晴れです。", "明日は雨？", "そうですね！", "以上"), JapaneseChunker().sentences("今日は晴れです。明日は雨？\nそうですね！以上"))
    }

    @Test fun chunkerRespectsLimitAndOverlap() {
        val chunks = JapaneseChunker(18, 1).chunk("一つ目の文です。二つ目の文です。三つ目の文です。四つ目の文です。")
        assertTrue(chunks.all { it.length <= 18 })
        assertTrue(chunks.size > 1)
        assertTrue(chunks[1].startsWith("二つ目の文です。"))
        assertEquals(listOf("あいうえお", "かきく"), JapaneseChunker(5, 0).chunk("あいうえおかきく"))
    }

    @Test fun indexSearchReplaceAndDimension() {
        val idx = VectorIndex()
        idx.add(IndexedChunk("a", "A", floatArrayOf(1f, 0f)))
        idx.add(IndexedChunk("b", "B", floatArrayOf(0f, 1f)))
        assertEquals("a", idx.search(floatArrayOf(1f, 0f), 1).first().chunk.id)
        idx.add(IndexedChunk("a", "A2", floatArrayOf(0f, 1f)))
        assertEquals(2, idx.size)
        assertFailsWith<IllegalArgumentException> { idx.add(IndexedChunk("c", "C", floatArrayOf(1f, 0f, 0f))) }
        assertFailsWith<IllegalArgumentException> { idx.search(floatArrayOf(1f, 0f, 0f), 1) }
    }

    @Test fun pipelineRetrievesAndReranks() = runTest {
        val e = CharEmbedder()
        val p = RagPipeline(e, e, reranker = { _, d -> if ("東京タワー" in d) 1f else 0f }, chunker = JapaneseChunker(40))
        p.ingest("d1", "東京タワーは港区にある電波塔です。高さは333メートルです。")
        p.ingest("d2", "瑠璃色は紫みを帯びた濃い青のことです。")
        val r = p.search("東京タワーはどこにある？", retrieveK = 5, topN = 2)
        assertTrue(r.first().chunk.id.startsWith("d1"))
        assertEquals(1f, r.first().rerankScore)
        val plain = RagPipeline(e, e)
        plain.ingest("d2", "瑠璃色は紫みを帯びた濃い青のことです。")
        assertNull(plain.search("瑠璃色", topN = 1).first().rerankScore)
    }
}
