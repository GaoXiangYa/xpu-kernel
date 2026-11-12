# xpu-kernel

CUDA kernel 练习工程：**每个算子一个目录，每个实现一个版本文件**，统一使用 GoogleTest 做正确性校验，并单独提供性能测试程序。

工程使用 **C++20**，依赖 CUDA Toolkit + CMake + GoogleTest。所有 kernel 的输入/输出统一为 **bf16**（CUDA `__nv_bfloat16`），内部用 fp32 计算，CPU 参考实现用 double 计算后再 round 回 bf16。已经内置 `gelu_mul` 作为完整示例。

## 特性

- 自动发现算子：顶层目录只要同时包含 `<name>/CMakeLists.txt` 和 `<name>/<name>.cuh` 就会参与构建，不需要修改根 `CMakeLists.txt`。
- 自动发现版本：`<name>_v*.cu` 会被自动加入测试和 benchmark 可执行文件，新增 `v1`、`v2` 时不需要改 CMake。
- 统一 bf16 数据通路：`xpu::bf16` 类型 + `bf16_from_float` / `bf16_to_float` 转换函数；kernel 内部用 fp32 计算，结果 round 回 bf16。
- GoogleTest 单元测试：CPU 端 double 精度参考实现 + 针对 bf16 精度调整的 `atol + rtol * |ref|` 组合容差；输出缓冲区预先填 NaN，漏写元素也会被发现。
- 无 GPU 环境自动跳过 device 测试（`GTEST_SKIP`），纯 CPU 的参考实现测试仍然会执行。
- 独立 benchmark：CUDA event 计时，支持 warmup / 多轮测量，输出 median、min、max。
- CTest 集成，方便一条命令跑完所有算子的单元测试。
- 配置好 VS Code + clangd：项目根目录的 `.clangd` 处理 nvcc 参数和 CUDA 13 头文件兼容问题，开箱即可跳转/补全。

## 目录结构

```text
xpu-kernel/
├── CMakeLists.txt              # 根构建脚本 + 算子自动发现
├── .clangd                     # clangd 的 CUDA 兼容配置
├── .vscode/
│   ├── settings.json           # clangd 路径、compile_commands、文件关联
│   └── extensions.json         # 推荐扩展
├── cmake/
│   └── KernelOp.cmake          # add_kernel_op()：构建单个算子
├── common/                     # 所有算子共用的头文件（header-only）
│   ├── bf16.cuh                # xpu::bf16 别名 + bf16 <-> fp32 转换
│   ├── cuda_check.cuh          # XPU_CUDA_CHECK / LAUNCH_CHECK / SYNC_CHECK
│   ├── cuda_test.h             # GoogleTest CUDA fixture（无 GPU 时 skip）
│   ├── device_buffer.cuh       # DeviceBuffer<T> RAII 显存管理
│   ├── test_utils.cuh          # 随机数据 + 带容差比较
│   ├── benchmark.cuh           # CUDA event benchmark
│   └── bench_options.h         # benchmark 命令行参数
├── gelu_mul/                   # 一个算子一个目录
│   ├── CMakeLists.txt          # add_kernel_op(gelu_mul)
│   ├── gelu_mul.cuh            # 接口、版本表、CPU 参考实现
│   ├── gelu_mul_v0.cu          # v0 实现 + launch 函数（nvcc 编译）
│   ├── gelu_mul_test.cpp       # GoogleTest 单元测试（host C++）
│   └── gelu_mul_bench.cpp      # benchmark 入口（host C++）
├── tools/clangd/cuda_stub/
│   ├── cuda_runtime.h          # 仅 clangd 使用的 CUDA 运行时声明子集
│   ├── cuda_bf16.h             # 仅 clangd 使用的 bf16 声明
│   └── cuda_math_shim.h        # 数学函数到 clang builtin 的映射
└── build/                      # 构建产物（git 忽略）
```

## 环境要求

- NVIDIA GPU，CUDA Toolkit >= 12（已在 CUDA 13.2 + CMake 3.28 上完成构建验证）
- CMake >= 3.24
- 支持 C++20 的编译器（工程中 C++ 和 CUDA 标准都设置为 C++20）
- GoogleTest：
  - 优先使用系统安装的 GoogleTest（本机为 `libgtest-dev`）；
  - 如果 `find_package(GTest)` 找不到，CMake 会通过 `FetchContent` 自动下载 `googletest v1.15.2`，此时需要网络。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

默认使用 `-DCMAKE_CUDA_ARCHITECTURES=native`，即为本机 GPU 生成代码。如果构建机器看不到 GPU（CI、远程编译机），显式指定架构：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86  # RTX 3060 Ti 是 sm_86
```

构建产物在 `build/bin/` 下：

- `build/bin/gelu_mul_test`：GoogleTest 单元测试
- `build/bin/gelu_mul_bench`：性能测试

CMake 会生成 `build/compile_commands.json`，供 clangd 使用。kernel 源文件（`*.cu`）由 nvcc 编译；测试和 benchmark（`*.cpp`）由 host C++ 编译器编译，因此 gtest 可以直接使用，不需要经过 nvcc。

## VS Code / clangd

项目已经包含 [`.vscode/settings.json`](.vscode/settings.json) 和 [`.clangd`](.clangd)。打开工程后安装推荐扩展即可：

1. 安装 `llvm-vs-code-extensions.vscode-clangd`（如果还没装）。
2. 用 CMake 配置一次，确保 `build/compile_commands.json` 存在：

   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   ```

3. VS Code 打开工程，clangd 会自动读取 `--compile-commands-dir=${workspaceFolder}/build` 并建立索引。

配置做了几件事：

- `.vscode/settings.json` 指定 clangd 路径、`compile_commands.json` 目录、后台索引、`cuda-cpp` 文件关联，并禁用 C/C++ 扩展的 IntelliSense 以避免冲突。
- `.clangd` 只对 `*.cu` / `*.cuh` 生效：把 nvcc 参数替换成 clang++ 能理解的参数，并移除 `-forward-unknown-to-host-compiler`、`--generate-code`、`-Xcompiler` 等 nvcc 专有选项。
- 本机 clangd 18 只支持到 CUDA 12.3，直接解析 CUDA 13.2 头文件会大量报错，因此用 `-nocudainc` + [`tools/clangd/cuda_stub/cuda_runtime.h`](tools/clangd/cuda_stub/cuda_runtime.h) / [`cuda_bf16.h`](tools/clangd/cuda_stub/cuda_bf16.h) 提供索引所需的 CUDA / bf16 声明子集。这个目录只给 clangd 使用，nvcc 不会看到它。
- 测试和 benchmark 是 `.cpp`，clangd 按普通 C++ 解析，gtest 补全和跳转正常。

维护提示：

- 如果之后换了 GPU 架构，同步修改 `.clangd` 里的 `--cuda-gpu-arch=sm_86`。
- 如果 CUDA 不在 `/usr/local/cuda`，修改 `.clangd` 里的 `--cuda-path`。
- 如果使用其他构建目录，需要同时修改 `.vscode/settings.json` 的 `--compile-commands-dir` 和 `.clangd` 中 `../tools/clangd/cuda_stub` 的相对路径（默认约定是 `<repo>/build`）。
- `clangd.path` 当前固定为 `/usr/bin/clangd`；换了 clangd 版本时改这一项即可。
- 如果新增 kernel 用到了 stub 里没有的 CUDA runtime API，在 [`tools/clangd/cuda_stub/cuda_runtime.h`](tools/clangd/cuda_stub/cuda_runtime.h) 里补声明；bf16 相关声明放在 [`cuda_bf16.h`](tools/clangd/cuda_stub/cuda_bf16.h)；常用数学函数已经在 [`cuda_math_shim.h`](tools/clangd/cuda_stub/cuda_math_shim.h) 中映射到 clang builtin。

用命令行检查 clangd 是否能正常解析某个文件：

```bash
clangd --check=gelu_mul/gelu_mul_v0.cu --compile-commands-dir=build --enable-config
```

## 运行单元测试

```bash
# 跑全部测试（CPU 参考测试 + GPU 正确性测试）
./build/bin/gelu_mul_test

# 只跑不依赖 GPU 的参考实现测试
./build/bin/gelu_mul_test --gtest_filter='GeluMulReference.*:GeluMul.NonPositiveSizeIsNoOp'

# 只跑 GPU kernel 正确性测试
./build/bin/gelu_mul_test --gtest_filter='GeluMulTest.*'

# 选择 GPU
CUDA_VISIBLE_DEVICES=1 ./build/bin/gelu_mul_test

# 通过 CTest 跑所有算子
ctest --test-dir build --output-on-failure
```

没有可见 GPU 时，使用 `xpu::test::CudaTest` fixture 的测试会显示 `SKIPPED`，退出码仍为 0。纯 CPU 测试不受影响。

测试内容：

- `GeluMulReference.KnownValues`：CPU 参考实现的已知 bf16 数值测试，不需要 GPU。
- `GeluMul.NonPositiveSizeIsNoOp`：验证 `n <= 0` 时 launch 函数直接返回，不需要 GPU。
- `GeluMulTest.AllVersionsMatchReference`：遍历所有 shape 和 `kVariants`，与 CPU double 参考实现比较；无 GPU 时 skip。

bf16 的尾数只有 8 bit，1 ULP 的相对误差大约在 `0.4% ~ 0.8%`。因此测试容差使用 `kRtol = 1e-2`、`kAtol = 1e-4`；kernel 内部用 fp32 计算后 round 回 bf16，和参考实现之间最多通常只差 1 个 bf16 ULP。

GTest 输出示例：

```text
[==========] Running 3 tests from 3 test suites.
[ RUN      ] GeluMulReference.KnownValues
[       OK ] GeluMulReference.KnownValues (0 ms)
[ RUN      ] GeluMul.NonPositiveSizeIsNoOp
[       OK ] GeluMul.NonPositiveSizeIsNoOp (0 ms)
[ RUN      ] GeluMulTest.AllVersionsMatchReference
[  SKIPPED ] GeluMulTest.AllVersionsMatchReference (1 ms)
[  PASSED  ] 2 tests.
[  SKIPPED ] 1 test
```

## 运行 benchmark

```bash
# 默认参数
./build/bin/gelu_mul_bench

# 自定义参数
./build/bin/gelu_mul_bench --warmup 50 --iters 200 --rounds 10

# 查看帮助
./build/bin/gelu_mul_bench --help
```

benchmark 参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--warmup N` | 计时前的预热 launch 次数 | 20 |
| `--iters N` | 每一轮的 launch 次数 | 100 |
| `--rounds N` | 计时轮数 | 5 |
| `--device N` | CUDA 设备编号 | 0 |
| `-h, --help` | 显示帮助 | - |

输出示例：

```text
== gelu_mul benchmark ==
out[i] = gelu_tanh(x[i]) * gate[i]  (bf16, fp32 math)
device 0: NVIDIA GeForce RTX 3060 Ti (sm_86, 8.0 GiB)
variants: v0_naive

n = 1000003
  v0_naive     bench median=0.0281 ms  min=0.0279  max=0.0290  (iters=100, rounds=5)
```

## 增加一个新版本

以 `gelu_mul` 为例，新增 `v1`（比如 bf16x2 向量化）：

1. 新建 `gelu_mul/gelu_mul_v1.cu`，实现内核和 `launch_v1`（签名与 `launch_v0` 一致）。
2. 在 `gelu_mul/gelu_mul.cuh` 中声明 `launch_v1`，并加入版本表：

   ```cpp
   inline constexpr Variant kVariants[] = {
       {"v0_naive", &launch_v0},
       {"v1_bf16x2", &launch_v1},
   };
   ```

3. 重新构建。`gelu_mul_v1.cu` 会被 CMake 自动扫描到，无需改 `CMakeLists.txt`。

版本函数统一签名：

```cpp
void launch_vN(const bf16* x, const bf16* gate, bf16* out, int n, cudaStream_t stream = nullptr);
```

约定：

- 输入/输出是长度为 `n` 的连续 bf16 张量，需支持 `n <= 0` 时直接返回。
- kernel 内部用 fp32 计算，写出前调用 `bf16_from_float` 转回 bf16。
- 使用 `stream` 启动内核，测试代码传入 `nullptr`（默认流）。
- launch 函数末尾调用 `XPU_CUDA_LAUNCH_CHECK()` 检查启动错误。
- 注册到 `kVariants` 后，现有测试会自动覆盖新版本，一般不需要改测试代码。

## 新增一个算子

1. 新建目录，例如 `rms_norm/`，放入 5 个文件（可直接复制 `gelu_mul/` 后重命名）：

   ```text
   rms_norm/rms_norm.cuh       # 接口、版本表、CPU 参考实现
   rms_norm/rms_norm_v0.cu     # 第一个实现
   rms_norm/rms_norm_test.cpp  # GoogleTest 单元测试（host C++）
   rms_norm/rms_norm_bench.cpp # benchmark（可选，host C++）
   rms_norm/CMakeLists.txt     # 内容：add_kernel_op(rms_norm)
   ```

2. 根目录重新执行 configure/build。CMake 会自动发现该目录（条件：`CMakeLists.txt` 和 `<目录名>.cuh` 都存在）。

测试文件里使用 `xpu::test::CudaTest` 作为 fixture 即可获得无 GPU 自动 skip 的行为：

```cpp
TEST_F(RmsNormTest, MatchesReference) {
  // ...
}
```

对于归约、矩阵乘等参数不同的算子，`launch_vN` 的签名可以自定义，但 `Variant::LaunchFn`、`common/cuda_test.h` 的使用方式和测试代码需要同步修改。`common/` 中的工具与算子无关，可以直接复用。

## 性能测试说明

- 每轮记录 `iters` 次 launch 的总时间再除以 `iters`，避免 `cudaEventRecord` 的固定开销污染单次测量；轮与轮之间取 median，min 作为较乐观的参考。
- 计时包含 kernel launch 开销。小尺寸（如 `n = 1`）的数值主要反映启动延迟，不要当作吞吐看。
- 同一个输入缓冲区会被反复使用，较小的张量可能命中 L2 cache，带宽会偏高。评估显存带宽时优先看较大的 `n`。
- 需要细粒度分析时可用 Nsight Compute：

  ```bash
  ncu --set full ./build/bin/gelu_mul_bench --warmup 0 --iters 1 --rounds 1
  ```

  构建默认开启 `-lineinfo`，方便在 ncu 中对应到源码行。不需要时可用 `-DXPU_ENABLE_LINEINFO=OFF` 关闭。

## gelu_mul 练习路线

- `v0`：一元素一线程的标量实现（已提供）。
- `v1`：`__nv_bfloat162` / 32-bit 向量化读写，处理尾部元素。
- `v2`：grid-stride loop，减少 block 数量并支持任意网格大小。
- `v3`：使用 `__tanhf` / 快速数学指令，并量化误差与性能的取舍。
- `v4`：128-bit 向量化（一次加载 8 个 bf16），配合 bf16x2 运算。
- `v5`：实现精确版 GELU（`erf`）并与 tanh 近似版对比。

## 代码格式化

```bash
clang-format -i common/*.cuh common/*.h gelu_mul/*.cu gelu_mul/*.cpp gelu_mul/*.cuh
```
