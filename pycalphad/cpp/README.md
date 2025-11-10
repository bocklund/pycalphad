# PyCalphad C++ Infrastructure

This directory contains the C++ implementation of performance-critical PyCalphad algorithms, with Cython bindings for Python integration.

## Directory Structure

```
pycalphad/cpp/
├── CMakeLists.txt          # CMake build configuration
├── README.md               # This file
├── include/                # C++ header files for core algorithms
│   ├── pycalphad_types.hpp # Common type definitions
│   └── (future module headers)
├── src/                    # C++ source files for core algorithms
│   └── (future module implementations)
└── utils/                  # Utility headers and implementations
    ├── numpy_wrapper.hpp   # NumPy array interface
    ├── numpy_wrapper.cpp
    ├── lapack_wrapper.hpp  # LAPACK function wrappers
    ├── lapack_wrapper.cpp
    ├── memory_utils.hpp    # RAII memory management
    └── memory_utils.cpp
```

## Build Configuration

### Compiler Requirements

- **C++ Standard**: C++17 or later
- **Compiler**: GCC 7+, Clang 5+, MSVC 2017+, or compatible

### Dependencies

- **NumPy**: For array interfacing (headers from Python package)
- **LAPACK/BLAS**: For linear algebra operations
  - Linux: `liblapack-dev`, `libblas-dev`
  - macOS: Uses Accelerate framework (built-in)
  - Windows: OpenBLAS recommended via conda/pip
- **SciPy**: For additional LAPACK functions (via Python)

### Compiler Flags

#### Debug Build
- `-g`: Enable debugging symbols
- `-O0`: No optimization
- `-DPYCALPHAD_DEBUG`: Enable bounds checking and assertions

#### Release Build
- `-O3`: Maximum optimization
- `-march=native`: Optimize for current CPU architecture
- `-DNDEBUG`: Disable assertions

#### Warnings
- `-Wall -Wextra -pedantic`: Enable comprehensive warnings
- MSVC: `/W4` for warning level 4

## Building

### With setuptools (integrated with Python build)

The C++ infrastructure is automatically built when you install pycalphad:

```bash
pip install -e .
```

This will:
1. Compile C++ utility sources
2. Link them with Cython extensions
3. Make headers available to Cython modules

### With CMake (standalone C++ library)

For C++ development and testing:

```bash
cd pycalphad/cpp
mkdir build
cd build
cmake ..
make
```

Optional: Build with tests:
```bash
cmake -DBUILD_TESTS=ON ..
make
ctest
```

## Migration Status

The migration from pure Cython to C++ with Cython bindings is in progress:

- [ ] **Phase 1**: Infrastructure setup ✅ (Current)
- [ ] **Phase 2**: Migrate `hyperplane` module
- [ ] **Phase 3**: Migrate `phase_rec` module
- [ ] **Phase 4**: Migrate `composition_set` module
- [ ] **Phase 5**: Migrate `eqsolver` module
- [ ] **Phase 6**: Migrate `minimizer` module

## Design Principles

### 1. Pure C++ Core
- Core algorithms implemented in pure C++ for portability
- No Python/Cython dependencies in C++ code
- Can be used in other projects or as standalone library

### 2. Thin Cython Bindings
- Cython `.pyx` files act as thin wrappers
- Handle Python object conversions
- Manage array lifetime and memory safety
- Translate C++ exceptions to Python exceptions

### 3. Type Safety
- Use C++17 features for type safety
- Template functions for generic array operations
- RAII for automatic resource management

### 4. Performance
- Zero-cost abstractions where possible
- Inline functions for performance-critical paths
- SIMD-friendly data layouts
- Minimize Python/C++ boundary crossings

### 5. Maintainability
- Clear separation of concerns
- Well-documented interfaces
- Comprehensive unit tests
- Consistent coding style

## Coding Conventions

### File Naming
- Headers: `*.hpp` (C++ header files)
- Sources: `*.cpp` (C++ implementation files)
- Cython wrappers: `*.pyx` (Cython source files)

### Namespaces
- All C++ code in `pycalphad` namespace
- Subnamespaces for modules: `pycalphad::hyperplane`, etc.
- Utility code in `pycalphad::numpy`, `pycalphad::lapack`, `pycalphad::memory`

### Naming Conventions
- Types: `PascalCase` (e.g., `PhaseRecord`)
- Functions: `snake_case` (e.g., `compute_equilibrium`)
- Constants: `UPPER_CASE` (e.g., `MIN_SITE_FRACTION`)
- Member variables: `snake_case_` with trailing underscore

### Documentation
- Doxygen-style comments for public APIs
- Explain algorithm choices and optimizations
- Document preconditions and postconditions

## Testing

C++ tests are separate from Python tests:

1. **C++ Unit Tests**: Test C++ functions in isolation (future)
2. **Integration Tests**: Test C++ via Cython bindings (existing Python tests)
3. **Performance Tests**: Benchmark against pure Cython implementation

## Contributing

When adding new C++ code:

1. Add header in `include/` with clear documentation
2. Add implementation in `src/` (if not header-only)
3. Update `CMakeLists.txt` to include new sources
4. Create Cython wrapper in `pycalphad/core/`
5. Add tests to verify correctness and performance
6. Update this README with migration status

## License

Same as PyCalphad: MIT License
