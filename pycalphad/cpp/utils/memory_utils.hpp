#ifndef PYCALPHAD_MEMORY_UTILS_HPP
#define PYCALPHAD_MEMORY_UTILS_HPP

#include "pycalphad_types.hpp"
#include <memory>
#include <cstdlib>
#include <cstring>

namespace pycalphad {
namespace memory {

/**
 * @brief RAII wrapper for malloc'd memory
 *
 * Provides automatic cleanup of dynamically allocated memory
 * using malloc/free (needed for compatibility with some C libraries)
 */
template<typename T>
class MallocPtr {
private:
    T* ptr_;
    size_type size_;

public:
    MallocPtr() : ptr_(nullptr), size_(0) {}

    explicit MallocPtr(size_type n) : ptr_(nullptr), size_(n) {
        if (n > 0) {
            ptr_ = static_cast<T*>(std::malloc(n * sizeof(T)));
            if (!ptr_) {
                throw std::bad_alloc();
            }
        }
    }

    ~MallocPtr() {
        if (ptr_) {
            std::free(ptr_);
        }
    }

    // Disable copy
    MallocPtr(const MallocPtr&) = delete;
    MallocPtr& operator=(const MallocPtr&) = delete;

    // Enable move
    MallocPtr(MallocPtr&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    MallocPtr& operator=(MallocPtr&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                std::free(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Accessors
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    T& operator[](size_type i) { return ptr_[i]; }
    const T& operator[](size_type i) const { return ptr_[i]; }
    size_type size() const { return size_; }

    // Reset to zero
    void zero() {
        if (ptr_ && size_ > 0) {
            std::memset(ptr_, 0, size_ * sizeof(T));
        }
    }

    // Release ownership
    T* release() {
        T* tmp = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        return tmp;
    }
};

/**
 * @brief RAII wrapper for 2D malloc'd memory
 *
 * Stores a contiguous block of memory as a 2D array
 */
template<typename T>
class MallocPtr2D {
private:
    T* ptr_;
    size_type rows_;
    size_type cols_;

public:
    MallocPtr2D() : ptr_(nullptr), rows_(0), cols_(0) {}

    MallocPtr2D(size_type rows, size_type cols)
        : ptr_(nullptr), rows_(rows), cols_(cols) {
        size_type total = rows * cols;
        if (total > 0) {
            ptr_ = static_cast<T*>(std::malloc(total * sizeof(T)));
            if (!ptr_) {
                throw std::bad_alloc();
            }
        }
    }

    ~MallocPtr2D() {
        if (ptr_) {
            std::free(ptr_);
        }
    }

    // Disable copy
    MallocPtr2D(const MallocPtr2D&) = delete;
    MallocPtr2D& operator=(const MallocPtr2D&) = delete;

    // Enable move
    MallocPtr2D(MallocPtr2D&& other) noexcept
        : ptr_(other.ptr_), rows_(other.rows_), cols_(other.cols_) {
        other.ptr_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
    }

    MallocPtr2D& operator=(MallocPtr2D&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                std::free(ptr_);
            }
            ptr_ = other.ptr_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            other.ptr_ = nullptr;
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }

    // Accessors (C-order: row-major)
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    T& operator()(size_type i, size_type j) {
        return ptr_[i * cols_ + j];
    }
    const T& operator()(size_type i, size_type j) const {
        return ptr_[i * cols_ + j];
    }

    size_type rows() const { return rows_; }
    size_type cols() const { return cols_; }
    size_type size() const { return rows_ * cols_; }

    // Reset to zero
    void zero() {
        if (ptr_ && size() > 0) {
            std::memset(ptr_, 0, size() * sizeof(T));
        }
    }

    // Release ownership
    T* release() {
        T* tmp = ptr_;
        ptr_ = nullptr;
        rows_ = 0;
        cols_ = 0;
        return tmp;
    }
};

/**
 * @brief Helper to allocate and zero-initialize array
 */
template<typename T>
inline MallocPtr<T> make_malloc_ptr_zeroed(size_type n) {
    MallocPtr<T> ptr(n);
    ptr.zero();
    return ptr;
}

template<typename T>
inline MallocPtr2D<T> make_malloc_ptr2d_zeroed(size_type rows, size_type cols) {
    MallocPtr2D<T> ptr(rows, cols);
    ptr.zero();
    return ptr;
}

} // namespace memory
} // namespace pycalphad

#endif // PYCALPHAD_MEMORY_UTILS_HPP
