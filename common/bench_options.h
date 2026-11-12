#pragma once

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace xpu::test {

struct BenchCliOptions {
  int warmup = 20;
  int iters = 100;
  int rounds = 5;
  int device = 0;
};

namespace detail {

inline void print_bench_usage(const char* program, const char* description) {
  if (description != nullptr) {
    std::printf("%s\n\n", description);
  }
  std::printf("Usage: %s [options]\n\n", program);
  std::printf("Options:\n");
  std::printf("  --warmup N   Untimed launches per benchmark (default: 20).\n");
  std::printf("  --iters N    Launches per timed round (default: 100).\n");
  std::printf("  --rounds N   Timed rounds per benchmark (default: 5).\n");
  std::printf("  --device N   CUDA device index (default: 0).\n");
  std::printf("  -h, --help   Show this help.\n");
}

inline int parse_int(const char* text, const char* flag, int minimum, const char* program) {
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(text, &end, 10);

  const bool valid = errno == 0 && end != text && *end == '\0' && value >= minimum && value <= INT_MAX;
  if (!valid) {
    std::fprintf(stderr, "error: %s expects an integer >= %d, got '%s'\n", flag, minimum, text);
    print_bench_usage(program, nullptr);
    std::exit(2);
  }
  return static_cast<int>(value);
}

inline const char* next_value(int argc, char** argv, int& index, const char* flag, const char* program) {
  if (index + 1 >= argc) {
    std::fprintf(stderr, "error: %s requires a value\n", flag);
    print_bench_usage(program, nullptr);
    std::exit(2);
  }
  return argv[++index];
}

}  // namespace detail

// Parses the command line of an operator benchmark binary. Exits on --help
// (status 0) or invalid input (status 2).
inline BenchCliOptions parse_bench_options(int argc, char** argv, const char* description = nullptr) {
  BenchCliOptions options;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];

    if (argument == "-h" || argument == "--help") {
      detail::print_bench_usage(argv[0], description);
      std::exit(0);
    } else if (argument == "--warmup") {
      const char* value = detail::next_value(argc, argv, i, "--warmup", argv[0]);
      options.warmup = detail::parse_int(value, "--warmup", 0, argv[0]);
    } else if (argument == "--iters") {
      const char* value = detail::next_value(argc, argv, i, "--iters", argv[0]);
      options.iters = detail::parse_int(value, "--iters", 1, argv[0]);
    } else if (argument == "--rounds") {
      const char* value = detail::next_value(argc, argv, i, "--rounds", argv[0]);
      options.rounds = detail::parse_int(value, "--rounds", 1, argv[0]);
    } else if (argument == "--device") {
      const char* value = detail::next_value(argc, argv, i, "--device", argv[0]);
      options.device = detail::parse_int(value, "--device", 0, argv[0]);
    } else {
      std::fprintf(stderr, "error: unknown option '%s'\n", argument.c_str());
      detail::print_bench_usage(argv[0], nullptr);
      std::exit(2);
    }
  }

  return options;
}

}  // namespace xpu::test
