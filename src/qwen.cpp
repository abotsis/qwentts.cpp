// qwen.cpp: public ABI implementation.
//
// Every entry declared in qwen.h lives here under one extern "C" block
// so the symbols carry C linkage and are linkable from C, Rust, Go,
// Python ctypes and any other binding generator. The struct
// qwen_context opaque handle owns one BackendPair, one PipelineTTS
// (which embeds its PipelineCodec) and one BPETokenizer. qwen_init
// walks the load chain in dependency order and unwinds whatever it
// already allocated when any step fails. qwen_free mirrors that order
// in reverse.
//
// This translation unit also absorbs the internal qt_set_error /
// qt_throw / qt_log helpers that the rest of the codebase calls. The
// internal qt_log_level enum is a typedef of the public qwen_log_level
// (same values, same layout) so a single log callback installed via
// qwen_log_set routes every diagnostic, internal or public.

#include "qwen.h"

#include "backend.h"
#include "bpe.h"
#include "pipeline-tts.h"
#include "qt-error.h"
#include "version.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

// Internal definition of the opaque handle. C++ types are fine here
// because nothing in this struct ever crosses the public ABI boundary :
// callers only ever see `struct qwen_context *`. PipelineTTS already
// embeds the PipelineCodec, so no separate codec field is needed.
struct qwen_context {
    BackendPair  bp;
    PipelineTTS  pt;
    BPETokenizer tok;
};

// Thread-local backing store for qt_last_error(). std::string sized once
// per thread, grows on demand, never freed across calls : the std runtime
// reclaims it on thread exit. An empty string means "no error recorded
// on this thread yet", which qt_last_error() exposes as "".
static thread_local std::string g_last_error;

void qt_set_error_v(const char * fmt, va_list ap) {
    if (!fmt) {
        g_last_error.clear();
        return;
    }
    // Two-pass vsnprintf : first call sizes the buffer, second writes the
    // message. va_copy keeps the original ap valid for the second pass.
    va_list ap2;
    va_copy(ap2, ap);
    int needed = std::vsnprintf(nullptr, 0, fmt, ap2);
    va_end(ap2);
    if (needed < 0) {
        g_last_error = "qt_set_error : vsnprintf failed";
        return;
    }
    g_last_error.resize(static_cast<size_t>(needed));
    std::vsnprintf(g_last_error.data(), static_cast<size_t>(needed) + 1, fmt, ap);
}

void qt_set_error(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    qt_set_error_v(fmt, ap);
    va_end(ap);
}

const char * qt_last_error(void) {
    return g_last_error.c_str();
}

// Formats a message with printf semantics and throws std::runtime_error.
// The catch site at the binary entry inspects the what() string and feeds
// it into qt_set_error so the user-visible diagnostic is identical
// whether the failure used the bool-return path or the throw path.
void qt_throw(const char * fmt, ...) {
    char buf[1024];
    if (fmt) {
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
    } else {
        buf[0] = '\0';
    }
    throw std::runtime_error(buf);
}

// Process-wide log callback. Atomic so qwen_log_set can replace it without
// locking : write happens with memory_order_release, every reader sees a
// fully published callback pointer paired with its user_data slot.
// std::atomic on a function pointer is lock-free on every platform we
// target. user_data is a plain pointer because it is only ever published
// alongside cb under the same release ordering.
static std::atomic<qwen_log_cb> g_log_cb{ nullptr };
static void *                   g_log_cb_user = nullptr;

// Routes one log line to the installed callback or to stderr. Two-pass
// vsnprintf sizes the heap buffer when the message exceeds the stack
// scratchpad, which keeps the common case allocation-free.
void qt_log(qt_log_level level, const char * fmt, ...) {
    if (!fmt) {
        return;
    }
    char    stackbuf[512];
    char *  buf    = stackbuf;
    int     needed = 0;
    va_list ap;
    va_start(ap, fmt);
    {
        va_list ap2;
        va_copy(ap2, ap);
        needed = std::vsnprintf(stackbuf, sizeof(stackbuf), fmt, ap2);
        va_end(ap2);
    }
    if (needed < 0) {
        va_end(ap);
        return;
    }
    std::string heapbuf;
    if ((size_t) needed >= sizeof(stackbuf)) {
        heapbuf.resize((size_t) needed);
        std::vsnprintf(heapbuf.data(), (size_t) needed + 1, fmt, ap);
        buf = heapbuf.data();
    }
    va_end(ap);

    qwen_log_cb cb = g_log_cb.load(std::memory_order_acquire);
    if (cb) {
        cb(level, buf, g_log_cb_user);
    } else {
        std::fprintf(stderr, "%s\n", buf);
    }
}

extern "C" {

const char * qwen_version(void) {
    // QWEN_VERSION is a string literal injected by tools/version.cmake
    // ("<git-hash> (<date>)"), so its storage already has process
    // lifetime and no formatting wrapper is needed.
    return QWEN_VERSION;
}

const char * qwen_last_error(void) {
    // c_str() on an empty std::string is guaranteed to point to a NUL
    // byte by C++11, so callers never have to NULL-check the result.
    return g_last_error.c_str();
}

void qwen_audio_free(struct qwen_audio * a) {
    if (!a) {
        return;
    }
    if (a->samples) {
        std::free(a->samples);
    }
    a->samples     = nullptr;
    a->n_samples   = 0;
    a->sample_rate = 0;
    a->channels    = 0;
}

void qwen_log_set(qwen_log_cb cb, void * user_data) {
    g_log_cb_user = user_data;
    g_log_cb.store(cb, std::memory_order_release);
}

void qwen_init_default_params(struct qwen_init_params * p) {
    p->abi_version = QWEN_ABI_VERSION;
    p->talker_path = nullptr;
    p->codec_path  = nullptr;
}

void qwen_tts_default_params(struct qwen_tts_params * p) {
    p->abi_version           = QWEN_ABI_VERSION;
    p->text                  = nullptr;
    p->lang                  = "english";
    p->instruct              = nullptr;
    p->speaker               = nullptr;
    p->ref_audio_24k         = nullptr;
    p->ref_n_samples         = 0;
    p->ref_text              = nullptr;
    p->seed                  = -1;
    p->max_new_tokens        = 2048;
    p->do_sample             = true;
    p->temperature           = 0.9f;
    p->top_k                 = 50;
    p->top_p                 = 1.0f;
    p->repetition_penalty    = 1.05f;
    p->subtalker_do_sample   = true;
    p->subtalker_temperature = 0.9f;
    p->subtalker_top_k       = 50;
    p->subtalker_top_p       = 1.0f;
    p->dump_dir              = nullptr;
}

struct qwen_context * qwen_init(const struct qwen_init_params * params) {
    if (!params || !params->talker_path || !params->codec_path) {
        qt_set_error("qwen_init : params, talker_path or codec_path is NULL");
        qt_log(QT_LOG_ERROR, "[qwen] qwen_init requires talker_path and codec_path");
        return nullptr;
    }
    if (params->abi_version > QWEN_ABI_VERSION) {
        qt_set_error(
            "qwen_init : params->abi_version %d > QWEN_ABI_VERSION %d (binding compiled against a newer header)",
            params->abi_version, QWEN_ABI_VERSION);
        qt_log(QT_LOG_ERROR, "[qwen] qwen_init params struct is from a newer ABI (%d > %d)", params->abi_version,
               QWEN_ABI_VERSION);
        return nullptr;
    }

    qt_log(QT_LOG_INFO, "[qwen] qwentts.cpp %s", qwen_version());

    // new qwen_context() value-initialises every field: POD aggregates
    // (BackendPair, PipelineTTS) are zero-init, std containers in
    // BPETokenizer construct empty.
    qwen_context * q = new qwen_context();

    // The load chain runs inside a try block. Any failure deep in the
    // GGUF reader, the codec load or the LM weight load throws via
    // qt_throw ; the catch funnels every variant into one cleanup via
    // qwen_free, which is idempotent on partial state (NULL-safe sched,
    // NULL GGUF handles, refcount-correct backend release).
    try {
        q->bp = backend_init("Talker");
        if (!q->bp.backend) {
            qt_throw("qwen_init : backend_init failed (no GGML backend available)");
        }

        if (!pipeline_tts_load(&q->pt, params->talker_path, params->codec_path, q->bp)) {
            qt_throw("qwen_init : pipeline_tts_load failed for '%s' / '%s'", params->talker_path, params->codec_path);
        }

        // BPE tokenizer payload lives inside the talker GGUF. Load the
        // base vocab + the qwen3-tts text specials in one shot. The
        // specials key list mirrors what the standalone CLI used to
        // do before the facade hoisted the load chain.
        if (!load_bpe_from_gguf(&q->tok, params->talker_path)) {
            qt_throw("qwen_init : load_bpe_from_gguf failed for '%s'", params->talker_path);
        }
        const char * specials_keys[] = {
            "qwen3-tts.text.im_start_id", "qwen3-tts.text.im_end_id",  "qwen3-tts.text.tts_pad_id",
            "qwen3-tts.text.tts_bos_id",  "qwen3-tts.text.tts_eos_id",
        };
        bpe_load_specials_from_keys(&q->tok, params->talker_path, specials_keys, 5);
    } catch (const std::exception & e) {
        qt_set_error("%s", e.what());
        qt_log(QT_LOG_ERROR, "[qwen] %s", e.what());
        qwen_free(q);
        return nullptr;
    }

    return q;
}

void qwen_free(struct qwen_context * q) {
    if (!q) {
        return;
    }
    pipeline_tts_free(&q->pt);
    backend_release(q->bp.backend, q->bp.cpu_backend);
    delete q;
}

// Resolve a -1 seed to a hardware random 64-bit value. Anything else is
// forwarded verbatim, so reproducibility is one explicit seed away.
static int64_t qwen_resolve_seed(int64_t seed) {
    if (seed >= 0) {
        return seed;
    }
    std::random_device rd;
    return (int64_t) (((uint64_t) rd() << 32) ^ (uint64_t) rd());
}

enum qwen_status qwen_synthesize(struct qwen_context *          q,
                                 const struct qwen_tts_params * params,
                                 struct qwen_audio *            out) {
    if (!q || !params || !out) {
        qt_set_error("qwen_synthesize : q, params or out is NULL");
        if (out) {
            qwen_audio_free(out);
        }
        return QWEN_STATUS_INVALID_PARAMS;
    }
    if (params->abi_version > QWEN_ABI_VERSION) {
        qt_set_error(
            "qwen_synthesize : params->abi_version %d > QWEN_ABI_VERSION %d (binding compiled against a newer header)",
            params->abi_version, QWEN_ABI_VERSION);
        qwen_audio_free(out);
        return QWEN_STATUS_INVALID_PARAMS;
    }

    // Mode validation. Mirrors the upstream Python which raises
    // ValueError when generate_voice_design is called on a non
    // voice_design model and the same shape applies to
    // generate_custom_voice. Explicit and KISS, so the caller never
    // gets a silently wrong synthesis. Messages preserved verbatim
    // from the previous CLI-side checks.
    const std::string & mt = q->pt.model_type;
    if (params->speaker && mt != "custom_voice") {
        qt_set_error("--speaker is only valid for custom_voice models (loaded: %s)", mt.c_str());
        qt_log(QT_LOG_ERROR, "[qwen] %s", qt_last_error());
        qwen_audio_free(out);
        return QWEN_STATUS_MODE_INVALID;
    }
    if (params->instruct && mt == "base") {
        qt_set_error("--instruct is not supported for base models");
        qt_log(QT_LOG_ERROR, "[qwen] %s", qt_last_error());
        qwen_audio_free(out);
        return QWEN_STATUS_MODE_INVALID;
    }
    if (mt == "custom_voice" && !params->speaker) {
        qt_set_error("custom_voice models require --speaker");
        qt_log(QT_LOG_ERROR, "[qwen] %s", qt_last_error());
        qwen_audio_free(out);
        return QWEN_STATUS_MODE_INVALID;
    }
    if (mt == "voice_design" && (!params->instruct || params->instruct[0] == '\0')) {
        qt_set_error("voice_design models require --instruct");
        qt_log(QT_LOG_ERROR, "[qwen] %s", qt_last_error());
        qwen_audio_free(out);
        return QWEN_STATUS_MODE_INVALID;
    }
    if (params->ref_audio_24k && mt != "base") {
        qt_set_error("--ref-audio is only valid for base models (loaded: %s)", mt.c_str());
        qt_log(QT_LOG_ERROR, "[qwen] %s", qt_last_error());
        qwen_audio_free(out);
        return QWEN_STATUS_MODE_INVALID;
    }
    if (params->speaker && params->ref_audio_24k) {
        qt_set_error("--speaker and --ref-audio are mutually exclusive");
        qt_log(QT_LOG_ERROR, "[qwen] %s", qt_last_error());
        qwen_audio_free(out);
        return QWEN_STATUS_INVALID_PARAMS;
    }
    if (params->ref_text && !params->ref_audio_24k) {
        qt_set_error("--ref-text requires --ref-audio");
        qt_log(QT_LOG_ERROR, "[qwen] %s", qt_last_error());
        qwen_audio_free(out);
        return QWEN_STATUS_INVALID_PARAMS;
    }

    // Translate the public POD params into the internal C++ struct
    // expected by pipeline_tts_synthesize. Borrowed pointers are
    // forwarded verbatim ; the lifetime contract on the public side
    // (caller keeps strings alive for the duration of the call)
    // matches what the pipeline already requires.
    PipelineTTSSynthesizeParams p = {};
    p.text                        = params->text;
    p.lang                        = params->lang;
    p.instruct                    = params->instruct;
    p.speaker                     = params->speaker;
    p.ref_audio_24k               = params->ref_audio_24k;
    p.ref_n_samples               = params->ref_n_samples;
    p.ref_text                    = params->ref_text;
    p.seed                        = qwen_resolve_seed(params->seed);
    p.max_new_tokens              = params->max_new_tokens;
    p.do_sample                   = params->do_sample;
    p.temperature                 = params->temperature;
    p.top_k                       = params->top_k;
    p.top_p                       = params->top_p;
    p.repetition_penalty          = params->repetition_penalty;
    p.subtalker_do_sample         = params->subtalker_do_sample;
    p.subtalker_temperature       = params->subtalker_temperature;
    p.subtalker_top_k             = params->subtalker_top_k;
    p.subtalker_top_p             = params->subtalker_top_p;
    p.dump_dir                    = params->dump_dir;

    // Defense in depth: the synthesis path normally reports failures
    // via bool return + qt_set_error. A future load-style throw or any
    // std::bad_alloc deep inside the GGML backend is caught here and
    // converted to QWEN_STATUS_GENERATE_FAILED so an exception never
    // crosses the extern "C" boundary.
    try {
        PipelineTTSSynthesizeOutput pout;
        if (!pipeline_tts_synthesize(&q->pt, &q->tok, p, &pout)) {
            qwen_audio_free(out);
            return QWEN_STATUS_GENERATE_FAILED;
        }

        // Copy the std::vector<float> into a malloc-backed buffer the
        // caller can free with std::free via qwen_audio_free. The
        // vector itself goes out of scope at function exit, releasing
        // its own storage independently.
        const size_t n     = pout.audio.size();
        const size_t bytes = n * sizeof(float);
        float *      buf   = (float *) std::malloc(bytes > 0 ? bytes : 1);
        if (!buf) {
            qt_set_error("qwen_synthesize : malloc failed for %zu samples", n);
            qt_log(QT_LOG_ERROR, "[qwen] %s", qt_last_error());
            qwen_audio_free(out);
            return QWEN_STATUS_OOM;
        }
        if (n > 0) {
            std::memcpy(buf, pout.audio.data(), bytes);
        }
        out->samples     = buf;
        out->n_samples   = (int) n;
        out->sample_rate = pout.sample_rate;
        out->channels    = 1;
        return QWEN_STATUS_OK;
    } catch (const std::exception & e) {
        qt_set_error("%s", e.what());
        qt_log(QT_LOG_ERROR, "[qwen] %s", e.what());
        qwen_audio_free(out);
        return QWEN_STATUS_GENERATE_FAILED;
    }
}

}  // extern "C"
