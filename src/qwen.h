#pragma once
// qwen.h: public ABI for qwentts.cpp.
//
// Single-header public API. Pure C99, consumable from C and C++ alike.
// Bindings (Python ctypes, Rust bindgen, Go cgo) parse this file directly.
// Style follows whisper.h / llama.h / omnivoice.h: extern "C" linkage on
// every entry, POD structs only, const char * UTF-8 strings, qwen_status
// enum returns.
//
// The opaque qwen_context handle aggregates every module the synthesis
// path needs (Talker LM weights, code predictor MTP head, optional
// speaker encoder, 12 Hz audio tokenizer codec, BPE tokenizer, GGML
// backend pair). One init, one free, one synthesize call covers the
// full TTS path. The lower-level pipeline_tts_* / pipeline_codec_*
// entries declared in pipeline-tts.h / pipeline-codec.h stay available
// for tooling that needs partial init, but they are intentionally not
// part of this public ABI.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Symbol visibility. Three Windows cases: building the SHARED target
// (QWENTTS_BUILD set, dllexport), consuming the SHARED target from
// outside (nothing set, dllimport), consuming the STATIC archive
// (QWENTTS_STATIC set by the static target's INTERFACE definitions,
// empty so the linker resolves the symbol directly without dllimport).
// On GCC/Clang the default-visibility attribute is harmless on static
// builds and required on shared builds.
#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(QWENTTS_STATIC)
#        define QWEN_API
#    elif defined(QWENTTS_BUILD)
#        define QWEN_API __declspec(dllexport)
#    else
#        define QWEN_API __declspec(dllimport)
#    endif
#elif defined(__GNUC__) || defined(__clang__)
#    define QWEN_API __attribute__((visibility("default")))
#else
#    define QWEN_API
#endif

// Struct ABI version. Incremented every time a public POD struct grows a
// new field at the end. Callers fill `.abi_version = QWEN_ABI_VERSION`
// (or let qwen_*_default_params set it). Entries that consume those
// structs reject inputs whose abi_version exceeds the build-time
// constant: this guards a binary built against vN from receiving a
// struct laid out for vN+1 by a freshly compiled binding. Adding fields
// stays backward compat because the new tail is zero init in older
// callers and the lib reads only what its abi_version permits.
//
// There is no separate semver triple. The runtime build identity is the
// git short hash + commit date string returned by qwen_version() ; for
// binding compat checks, QWEN_ABI_VERSION is the only number that
// matters. Aligned on OV_ABI_VERSION = 2 for the omnivoice ABI cousin.
#define QWEN_ABI_VERSION 2

// Returns a static string of the form "<git-hash> (<date>)" identifying
// the exact commit this binary was built from. Safe to call from any
// thread, no allocation. Pointer stays valid for the process lifetime.
QWEN_API const char * qwen_version(void);

// Status code returned by every fallible entry. QWEN_STATUS_OK is always
// zero so `if (rc)` reads as `if (rc != QWEN_STATUS_OK)`.
enum qwen_status {
    QWEN_STATUS_OK              = 0,
    QWEN_STATUS_INVALID_PARAMS  = -1,
    QWEN_STATUS_MODE_INVALID    = -2,
    QWEN_STATUS_GENERATE_FAILED = -3,
    QWEN_STATUS_OOM             = -4,
};

// Returns the last error message produced on the calling thread by any
// qwen_* entry, as a NUL terminated UTF-8 string. errno-style semantics:
// the pointer is only meaningful right after a failure (qwen_init
// returning NULL, or any qwen_* entry returning a negative qwen_status) ;
// calling it after a successful entry yields the previous message or an
// empty string. Storage is thread local so two threads running
// qwen_synthesize concurrently never race on each other's diagnostics.
// The pointer stays valid until the next failing qwen_* entry on the
// same thread.
QWEN_API const char * qwen_last_error(void);

// Output audio buffer. Plain POD: the samples pointer is malloc
// allocated by qwen_synthesize, owned by the struct, released by
// qwen_audio_free. Do not free samples directly nor reassign without
// freeing first. Zero initialise before the first use:
// `struct qwen_audio a = {0};`.
struct qwen_audio {
    float * samples;      // mono PCM, malloc allocated
    int     n_samples;    // length in samples
    int     sample_rate;  // 24000 for the 12 Hz Qwen3-TTS tokenizer
    int     channels;     // 1 (mono)
};

// Release the samples buffer and reset the struct to empty. Safe on a
// zero initialised struct (no double free, no NULL deref).
QWEN_API void qwen_audio_free(struct qwen_audio * a);

// Opaque handle. Definition lives in qwen.cpp. Use qwen_init / qwen_free.
struct qwen_context;

// Initialisation parameters. Both GGUF paths are required: the talker
// GGUF holds the LM weights, the code predictor MTP head and (for
// custom_voice / voice_design checkpoints) the speaker encoder ; the
// codec GGUF holds the 12 Hz audio tokenizer. abi_version stays first
// so a future struct growth keeps reading the version field at offset
// 0. No use_fa / clamp_fp16 yet : the current pipeline_tts_load picks
// flash attention from backend capability without a user knob.
struct qwen_init_params {
    int          abi_version;
    const char * talker_path;
    const char * codec_path;
};

// Initialise to the standard defaults: both paths NULL (caller must set
// them before calling qwen_init), abi_version set to QWEN_ABI_VERSION.
QWEN_API void qwen_init_default_params(struct qwen_init_params * p);

// Allocate every module described by params. Returns NULL on any
// failure after releasing whatever it has allocated so far. The
// returned handle owns its GGML backend pair and must be released with
// qwen_free.
QWEN_API struct qwen_context * qwen_init(const struct qwen_init_params * params);

// Release every module owned by the handle and free the handle itself.
// Safe on NULL.
QWEN_API void qwen_free(struct qwen_context * q);

// Log severity. Numerically ordered so a callback can filter with a
// simple `if (level < threshold) return;`. ERROR is reserved for
// failure reports that the lib also surfaces via qwen_status /
// qwen_last_error ; WARN for recoverable surprises ; INFO for the
// normal load and synthesis cadence ; DEBUG for tensor-level cossim
// diagnostics.
enum qwen_log_level {
    QWEN_LOG_DEBUG = 0,
    QWEN_LOG_INFO  = 1,
    QWEN_LOG_WARN  = 2,
    QWEN_LOG_ERROR = 3,
};

// Logging callback. msg is a NUL terminated UTF-8 string already
// formatted by the lib, with no trailing newline (the callback is free
// to add one). user_data is forwarded verbatim from qwen_log_set.
// Called from any thread the lib runs on: the callback must be
// reentrant.
typedef void (*qwen_log_cb)(enum qwen_log_level level, const char * msg, void * user_data);

// Install a global log callback. Passing cb == NULL restores the
// default behaviour (write to stderr). Safe to call at any point ;
// takes effect immediately on subsequent log emissions across every
// thread. Storage is process wide, not per handle, matching
// whisper_log_set / llama_log_set / ov_log_set.
QWEN_API void qwen_log_set(qwen_log_cb cb, void * user_data);

// Synthesis parameters. Strings are NULL terminated UTF-8 ; NULL maps
// to empty where the underlying pipeline accepts it. The selection
// between base / custom_voice / voice_design synthesis mode is driven
// by the model_type read from the talker GGUF at qwen_init time, not
// by an explicit flag here ; the seven mode rules are enforced inside
// qwen_synthesize and surface as QWEN_STATUS_MODE_INVALID with a
// descriptive qwen_last_error(). abi_version stays first so the lib
// can route on it before reading any field that may have shifted in a
// future minor.
struct qwen_tts_params {
    int abi_version;

    // Input text and language hint. lang accepts the upstream
    // qwen3-tts language names ("english", "chinese", "auto", ...).
    // instruct is the style instruction string ; required for
    // voice_design, optional for custom_voice, rejected for base.
    // speaker is the named speaker for custom_voice models, rejected
    // for the other two modes.
    const char * text;
    const char * lang;
    const char * instruct;
    const char * speaker;

    // Optional voice reference for base mode voice cloning. Mode A
    // (x_vector_only) sets ref_audio_24k only ; mode B (ICL) sets
    // both ref_audio_24k and ref_text. ref_audio_24k is a mono float
    // PCM buffer sampled at the codec sample rate (24 kHz). Mutually
    // exclusive with speaker. Rejected for custom_voice / voice_design.
    const float * ref_audio_24k;
    int           ref_n_samples;
    const char *  ref_text;

    // Sampling configuration. seed == -1 is resolved by qwen_synthesize
    // to a hardware random seed via std::random_device, anything else
    // is forwarded verbatim for deterministic replay across runs.
    // Defaults match the upstream Python reference: do_sample true,
    // temperature 0.9, top_k 50, top_p 1.0, repetition_penalty 1.05,
    // subtalker mirrors talker, max_new_tokens 2048.
    int64_t seed;
    int     max_new_tokens;
    bool    do_sample;
    float   temperature;
    int     top_k;
    float   top_p;
    float   repetition_penalty;
    bool    subtalker_do_sample;
    float   subtalker_temperature;
    int     subtalker_top_k;
    float   subtalker_top_p;

    // Intermediate tensor dump directory. NULL disables dumps. Debug
    // only, slows the run.
    const char * dump_dir;
};

// Initialise to the standard defaults. Strings NULL, seed -1,
// max_new_tokens 2048, do_sample true, temperature 0.9, top_k 50,
// top_p 1.0, repetition_penalty 1.05, subtalker mirrors talker,
// dump_dir NULL.
QWEN_API void qwen_tts_default_params(struct qwen_tts_params * p);

// Run the full TTS synthesis. Validates the params against the loaded
// model_type (the seven base / custom_voice / voice_design rules),
// resolves the seed, hands off to pipeline_tts_synthesize and fills
// `out` with mono float PCM at the codec sample rate. Returns
// QWEN_STATUS_OK on success ; on any failure returns a negative
// qwen_status describing the cause and leaves `out` empty.
QWEN_API enum qwen_status qwen_synthesize(struct qwen_context *          q,
                                          const struct qwen_tts_params * params,
                                          struct qwen_audio *            out);

#ifdef __cplusplus
}
#endif
