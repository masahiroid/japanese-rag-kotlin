package jp.masahirocom.ragkit.litert

import jp.masahirocom.ragkit.Reranker
import jp.masahirocom.ragkit.SentenceEmbedder
import jp.masahirocom.ragkit.UnigramTokenizer
import jp.masahirocom.ragkit.VectorMath
import java.io.File
import java.io.FileInputStream
import java.nio.channels.FileChannel
import java.nio.MappedByteBuffer
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.tensorflow.lite.Interpreter

/** Memory-maps a `.tflite` file (e.g. one downloaded from masahiroid/ruri-v3-*-tflite). */
fun mapModel(file: File): MappedByteBuffer =
    FileInputStream(file).use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length()) }

/**
 * Shared plumbing: the converted graphs take `input_ids` and `attention_mask` as int32 `[1, seq]` (in that order).
 * One interpreter is not thread-safe, so calls are serialized.
 */
abstract class LiteRtModel(model: MappedByteBuffer, protected val sequenceLength: Int, numThreads: Int) : AutoCloseable {
    private val interpreter = Interpreter(model, Interpreter.Options().setNumThreads(numThreads))
    private val lock = Mutex()

    protected suspend fun run(ids: IntArray, mask: IntArray, outputSize: Int): FloatArray = lock.withLock {
        val out = Array(1) { FloatArray(outputSize) }
        interpreter.runForMultipleInputsOutputs(
            arrayOf<Any>(arrayOf(ids), arrayOf(mask)),
            mapOf<Int, Any>(0 to out),
        )
        out[0]
    }

    override fun close() = interpreter.close()
}

/** ruri-v3 embedding model (`ruri-v3-*_seqN.tflite`): mean pooling + L2 norm are inside the graph. */
class LiteRtEmbedder(
    model: MappedByteBuffer,
    private val tokenizer: UnigramTokenizer,
    sequenceLength: Int = 128,
    private val dimension: Int,
    numThreads: Int = 4,
) : LiteRtModel(model, sequenceLength, numThreads), SentenceEmbedder {
    override suspend fun embed(text: String): FloatArray {
        val (ids, mask) = tokenizer.padToLength(tokenizer.encode(text, sequenceLength), sequenceLength)
        return run(ids, mask, dimension)
    }
}

/** Japanese cross-encoder reranker (`japanese-reranker-*_seqN.tflite`, `ruri-v3-reranker-*`): relevance logit -> sigmoid. */
class LiteRtReranker(
    model: MappedByteBuffer,
    private val tokenizer: UnigramTokenizer,
    sequenceLength: Int = 256,
    private val applySigmoid: Boolean = true,
    numThreads: Int = 4,
) : LiteRtModel(model, sequenceLength, numThreads), Reranker {
    override suspend fun score(query: String, document: String): Float {
        val (ids, mask) = tokenizer.padToLength(tokenizer.encodePair(query, document, sequenceLength), sequenceLength)
        val logit = run(ids, mask, 1)[0]
        return if (applySigmoid) VectorMath.sigmoid(logit) else logit
    }
}
