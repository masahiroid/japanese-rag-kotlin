package jp.masahirocom.ragkit

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class TokenizerFixtureTest {
    private val tokenizer by lazy { UnigramTokenizer.fromJson(File("../tokenizers/ruri-v3/tokenizer.json").readText()) }
    private val fixtures by lazy {
        Json.parseToJsonElement(javaClass.getResource("/tokenizer_fixtures.json")!!.readText()).jsonObject
    }

    @Test
    fun singleTextsMatchHuggingFaceIds() {
        val mismatches = mutableListOf<String>()
        val items = fixtures["single"]!!.jsonArray
        for (e in items) {
            val o = e.jsonObject
            val text = o["text"]!!.jsonPrimitive.content
            val want = o["ids"]!!.jsonArray.map { it.jsonPrimitive.int }
            val got = tokenizer.encode(text).toList()
            if (want != got) mismatches += "${text.take(40).replace("\n", "\\n")} want=${want.take(12)} got=${got.take(12)}"
        }
        println("single: ${items.size - mismatches.size}/${items.size} exact")
        assertTrue(mismatches.isEmpty(), "mismatches (${mismatches.size}):\n" + mismatches.take(10).joinToString("\n"))
    }

    @Test
    fun pairsMatchHuggingFaceIds() {
        for (e in fixtures["pair"]!!.jsonArray) {
            val o = e.jsonObject
            val want = o["ids"]!!.jsonArray.map { it.jsonPrimitive.int }
            val got = tokenizer.encodePair(o["a"]!!.jsonPrimitive.content, o["b"]!!.jsonPrimitive.content).toList()
            assertEquals(want, got, "pair ${o["a"]} / ${o["b"]}")
        }
    }

    @Test
    fun paddingAndTruncation() {
        val ids = tokenizer.encode("検索文書: 今日は晴れです。")
        val (padded, mask) = tokenizer.padToLength(ids, 128)
        assertEquals(128, padded.size)
        assertEquals(ids.size, mask.sum())
        assertEquals(ids.toList(), padded.take(ids.size))
        assertEquals(6, tokenizer.encode("あいうえおかきくけこ", maxLength = 6).size)
        assertEquals(tokenizer.eosId, tokenizer.encode("あいうえおかきくけこ", maxLength = 6).last())
    }
}
