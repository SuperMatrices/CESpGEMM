#pragma once
#include <iostream>
#include <sstream>
#include <mutex>
#include <chrono>
#include <thread>
#include <atomic>
#include <iomanip>

// ======================
// Compile-time debug level
// ======================
//
// Example:
//   g++ ... -DDEBUG_LEVEL=3
//
// DEBUG_LEVEL = 0 → no debug logs compiled
// DEBUG_LEVEL = 1 → minimal debug
// DEBUG_LEVEL = 5 → very verbose
//
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

// ======================
// Runtime debug level
// ======================
namespace Logger {
    inline std::atomic<int> runtime_debug_level{0};
    inline std::mutex log_mutex;

    inline void set_runtime_debug_level(int lvl) {
        runtime_debug_level.store(lvl, std::memory_order_relaxed);
    }

    // timestamp string
    inline std::string timestamp() {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto t = system_clock::to_time_t(now);
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&t), "%F %T")
           << "." << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    // thread id string
    inline std::string thread_id_str() {
        std::stringstream ss;
        ss <<"Thread " << (void*)(pthread_self());
        return ss.str();
    }

    inline void log(const std::string& level,
                    const std::string& file,
                    int line,
                    const std::string& msg)
    {
        std::lock_guard<std::mutex> guard(log_mutex);
        std::cerr << "[" << timestamp()
                  << "][" << thread_id_str()
                  << "][" << level
                  << "][" << file << ":" << line << "] "
                  << msg << std::endl;
    }
}



// ======================
// Log level macros
// ======================

// -------- INFO --------
#define LOG_INFO(msg) \
    do { \
        std::stringstream _ss; _ss << msg; \
        Logger::log("INFO", __FILE__, __LINE__, _ss.str()); \
    } while (0)

// -------- WARN --------
#define LOG_WARN(msg) \
    do { \
        std::stringstream _ss; _ss << msg; \
        Logger::log("WARN", __FILE__, __LINE__, _ss.str()); \
    } while (0)

// -------- ERROR --------
#define LOG_ERROR(msg) \
    do { \
        std::stringstream _ss; _ss << msg; \
        Logger::log("ERROR", __FILE__, __LINE__, _ss.str()); \
    } while (0)


// ======================
// Debug logging (compile-time + runtime)
// ======================
#if DEBUG_LEVEL > 0

#define LOG_DEBUG(level, msg) \
    do { \
        if ((level) <= DEBUG_LEVEL && (level) <= Logger::runtime_debug_level.load()) { \
            std::stringstream _ss; _ss << msg; \
            Logger::log(std::string("DEBUG") + std::to_string(level), \
                        __FILE__, __LINE__, _ss.str()); \
        } \
    } while (0)

#else
// debug removed entirely
#define LOG_DEBUG(level, msg) do {} while(0)

#endif
