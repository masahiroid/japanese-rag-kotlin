package jp.masahirocom.ragkit.sample

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import jp.masahirocom.ragkit.PrefixedEmbedder
import jp.masahirocom.ragkit.RagPipeline
import jp.masahirocom.ragkit.RuriPromptPrefix
import jp.masahirocom.ragkit.UnigramTokenizer
import jp.masahirocom.ragkit.litert.LiteRtEmbedder
import jp.masahirocom.ragkit.litert.LiteRtReranker
import jp.masahirocom.ragkit.litert.mapModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Minimal on-device check: tokenizes, embeds and retrieves with the converted ruri-v3 `.tflite`, optionally reranks.
 * Models are read from the app's external files dir (adb push to /sdcard/Android/data/<package>/files/):
 *   ruri-v3-30m_seq128.tflite (required, dim 256) and japanese-reranker-xsmall-v2_seq256.tflite (optional).
 * Results go to the screen and to logcat (tag RAGSAMPLE).
 */
class MainActivity : Activity() {
    private val tag = "RAGSAMPLE"
    private lateinit var out: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        out = TextView(this).apply { textSize = 13f; setPadding(24, 24, 24, 24) }
        setContentView(ScrollView(this).apply { addView(out) })
        CoroutineScope(Dispatchers.Default).launch { runCheck() }
    }

    private suspend fun say(line: String) {
        Log.i(tag, line)
        withContext(Dispatchers.Main) { out.append(line + "\n") }
    }

    private suspend fun runCheck() {
        try {
            val dir = getExternalFilesDir(null)!!
            val tokenizer = UnigramTokenizer.fromJson(assets.open("tokenizer.json").bufferedReader().readText())
            say("tokenizer ok; 検索クエリ: 東京タワー -> " + tokenizer.encode("検索クエリ: 東京タワー").take(8))
            val embFile = File(dir, "ruri-v3-30m_seq128.tflite")
            if (!embFile.exists()) { say("MISSING $embFile"); return }
            val t0 = System.nanoTime()
            val base = LiteRtEmbedder(mapModel(embFile), tokenizer, sequenceLength = 128, dimension = 256)
            say("embedder loaded in ${(System.nanoTime() - t0) / 1_000_000} ms")

            val rag = RagPipeline(PrefixedEmbedder(base, RuriPromptPrefix.SEARCH_QUERY), PrefixedEmbedder(base, RuriPromptPrefix.SEARCH_DOCUMENT))
            val docs = mapOf(
                "tower" to "東京タワーは港区にある電波塔で、高さは333メートルです。",
                "ruri" to "瑠璃色は、紫みを帯びた濃い青のことである。",
                "fuji" to "富士山は静岡県と山梨県にまたがる日本一高い山です。",
            )
            val t1 = System.nanoTime()
            docs.forEach { (id, text) -> rag.ingest(id, text) }
            say("ingested ${docs.size} docs in ${(System.nanoTime() - t1) / 1_000_000} ms")
            var correct = 0
            for ((q, want) in listOf("日本で一番高い山は？" to "fuji", "紫がかった青い色" to "ruri", "港区にある電波塔" to "tower")) {
                val t = System.nanoTime()
                val r = rag.search(q, retrieveK = 3, topN = 3)
                val ms = (System.nanoTime() - t) / 1_000_000
                val top = r.first().chunk.id.substringBefore('#')
                if (top == want) correct++
                say("$q -> " + r.joinToString("  ") { "${it.chunk.id.substringBefore('#')}:%.3f".format(it.retrievalScore) } + "   (${ms} ms, top=${if (top == want) "OK" else "NG"})")
            }
            say("RETRIEVAL $correct/3 correct")

            val rrFile = File(dir, "japanese-reranker-xsmall-v2_seq256.tflite")
            if (rrFile.exists()) {
                val rr = LiteRtReranker(mapModel(rrFile), tokenizer, sequenceLength = 256)
                val q = "東京タワーはどこにある？"
                val scored = docs.map { (id, text) -> id to rr.score(q, text) }.sortedByDescending { it.second }
                say("RERANK $q -> " + scored.joinToString("  ") { "${it.first}:%.3f".format(it.second) })
            } else say("reranker file not present (optional)")
            say("DONE")
        } catch (e: Throwable) {
            say("ERROR ${e::class.java.simpleName}: ${e.message}")
            Log.e(tag, "failed", e)
        }
    }
}
