#pragma once
// qt-error.h : internal helpers backing the public qwen_last_error
// entry and the qwen_log callback routing.
//
// Not part of the public ABI. Translation units that emit user-facing
// errors include this header to record a diagnostic on the calling
// thread before they return a negative qwen_status (or false). The
// actual storage and the public qwen_last_error() reader live in
// qwen.cpp.
//
// Storage is thread_local so concurrent qwen_synthesize calls on
// different threads never race on each other's messages. The setter is
// variadic with printf semantics ; messages longer than the internal
// buffer are truncated, never split. Passing NULL as fmt clears the
// slot.
//
// qt_throw is the load-path counterpart : functions deep inside the
// GGUF reader and the codec load chain cannot return false up dozens
// of call sites without a massive cascade. They throw a
// std::runtime_error instead, which the ABI boundary entries
// (qwen_init, qwen_synthesize) catch and convert into qt_set_error
// plus a negative qwen_status. Exceptions never cross any future C ABI.
//
// qt_log routes a formatted message to the user-installed qwen_log_cb,
// or to stderr when no callback is installed. Used by every translation
// unit in the lib that wants its diagnostics to be redirectable from a
// wrapper (Python logging, Rust tracing, ...). The level enum is the
// public qwen_log_level, re-exported here under the historic qt_log_level
// name so existing call sites stay pixel perfect.

#include "qwen.h"

#include <cstdarg>

// Internal log level alias. Same values, same layout as the public
// qwen_log_level enum : a single underlying type means a single log
// callback installed through qwen_log_set routes every diagnostic
// without any cast or translation.
typedef enum qwen_log_level qt_log_level;

#define QT_LOG_DEBUG QWEN_LOG_DEBUG
#define QT_LOG_INFO  QWEN_LOG_INFO
#define QT_LOG_WARN  QWEN_LOG_WARN
#define QT_LOG_ERROR QWEN_LOG_ERROR

void qt_set_error(const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

void qt_set_error_v(const char * fmt, va_list ap);

// Throws std::runtime_error formatted with printf semantics. Tagged
// noreturn so the compiler can prune unreachable branches at the call
// site. Designed for the GGUF / codec load path where any failure means
// the model is unusable and unwinding to the boundary is the only sane
// recovery.
[[noreturn]] void qt_throw(const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

// Routes a formatted message at the requested level to the installed
// callback, or to stderr when none is set. The message is the full
// line without trailing newline ; routing layers add their own framing.
void qt_log(qt_log_level level, const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

// Returns the most recent error message recorded on the calling thread.
// Returns "" if no error has been set on this thread. The pointer stays
// valid until the next qt_set_error call on the same thread.
const char * qt_last_error(void);
