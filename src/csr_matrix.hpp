// CSR Matrix implementation per assignment spec
#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <vector>
#include <exception>

namespace sjtu {

class size_mismatch : public std::exception {
public:
    const char *what() const noexcept override {
        return "Size mismatch";
    }
};

class invalid_index : public std::exception {
public:
    const char *what() const noexcept override {
        return "Index out of range";
    }
};

// CSR matrix supporting get/set and basic operations
// Only std::vector is used per requirements
template <typename T>
class CSRMatrix {

private:
    size_t n_rows{};
    size_t n_cols{};
    std::vector<size_t> indptr_;   // size n_rows + 1
    std::vector<size_t> indices_;  // column indices, length = nnz
    std::vector<T> data_;          // values, length = nnz

    // binary search for column j within row
    size_t find_in_row(size_t row, size_t j) const {
        size_t l = indptr_[row];
        size_t r = indptr_[row + 1];
        while (l < r) {
            size_t mid = (l + r) >> 1;
            size_t cj = indices_[mid];
            if (cj == j) return mid;
            if (cj < j) l = mid + 1; else r = mid;
        }
        return static_cast<size_t>(-1);
    }

    // lower_bound position to keep indices sorted
    size_t lower_bound_in_row(size_t row, size_t j) const {
        size_t l = indptr_[row];
        size_t r = indptr_[row + 1];
        while (l < r) {
            size_t mid = (l + r) >> 1;
            if (indices_[mid] < j) l = mid + 1; else r = mid;
        }
        return l;
    }

public:
    // Assignment operators are deleted
    CSRMatrix &operator=(const CSRMatrix &other) = delete;
    CSRMatrix &operator=(CSRMatrix &&other) = delete;

    // Copy/move constructors
    CSRMatrix(const CSRMatrix &other) = default;
    CSRMatrix(CSRMatrix &&other) = default;

    // Constructor for empty matrix with dimensions
    CSRMatrix(size_t n, size_t m)
        : n_rows(n), n_cols(m), indptr_(n + 1, 0) {}

    // Constructor with pre-built CSR components
    CSRMatrix(size_t n, size_t m, size_t count,
              const std::vector<size_t> &indptr,
              const std::vector<size_t> &indices,
              const std::vector<T> &data)
        : n_rows(n), n_cols(m), indptr_(indptr), indices_(indices), data_(data) {
        if (indptr_.size() != n_rows + 1) throw size_mismatch();
        if (indices_.size() != count || data_.size() != count) throw size_mismatch();
        if (!indptr_.empty() && indptr_.front() != 0) throw size_mismatch();
        if (indptr_.back() != count) throw size_mismatch();
        // Validate per-row sorted indices and bounds
        for (size_t i = 0; i < n_rows; ++i) {
            size_t s = indptr_[i], e = indptr_[i + 1];
            for (size_t k = s; k < e; ++k) {
                if (indices_[k] >= n_cols) throw invalid_index();
                if (k + 1 < e && indices_[k] >= indices_[k + 1]) throw size_mismatch();
            }
        }
    }

    // Constructor from dense matrix (vector of vectors)
    CSRMatrix(size_t n, size_t m, const std::vector<std::vector<T>> &mat)
        : n_rows(n), n_cols(m) {
        indptr_.resize(n_rows + 1);
        indptr_[0] = 0;
        for (size_t i = 0; i < n_rows; ++i) {
            size_t cnt = 0;
            for (size_t j = 0; j < n_cols; ++j) {
                if (mat[i][j] != T()) {
                    indices_.push_back(j);
                    data_.push_back(mat[i][j]);
                    ++cnt;
                }
            }
            indptr_[i + 1] = indptr_[i] + cnt;
        }
    }

    ~CSRMatrix() = default;

    // Get dimensions and non-zero count
    size_t getRowSize() const { return n_rows; }
    size_t getColSize() const { return n_cols; }
    size_t getNonZeroCount() const { return data_.size(); }

    // Element access
    T get(size_t i, size_t j) const {
        if (i >= n_rows || j >= n_cols) throw invalid_index();
        size_t pos = find_in_row(i, j);
        if (pos == static_cast<size_t>(-1)) return T();
        return data_[pos];
    }

    void set(size_t i, size_t j, const T &value) {
        if (i >= n_rows || j >= n_cols) throw invalid_index();
        size_t pos = find_in_row(i, j);
        if (pos != static_cast<size_t>(-1)) {
            data_[pos] = value;
            return;
        }
        // Insert new entry keeping row sorted
        size_t insert_pos = lower_bound_in_row(i, j);
        indices_.insert(indices_.begin() + insert_pos, j);
        data_.insert(data_.begin() + insert_pos, value);
        for (size_t r = i + 1; r < indptr_.size(); ++r) ++indptr_[r];
    }

    // Access CSR components
    const std::vector<size_t> &getIndptr() const { return indptr_; }
    const std::vector<size_t> &getIndices() const { return indices_; }
    const std::vector<T> &getData() const { return data_; }

    // Convert to dense matrix format
    std::vector<std::vector<T>> getMatrix() const {
        std::vector<std::vector<T>> mat(n_rows, std::vector<T>(n_cols, T()));
        for (size_t i = 0; i < n_rows; ++i) {
            for (size_t k = indptr_[i]; k < indptr_[i + 1]; ++k) {
                mat[i][indices_[k]] = data_[k];
            }
        }
        return mat;
    }

    // Matrix-vector multiplication
    std::vector<T> operator*(const std::vector<T> &vec) const {
        if (vec.size() != n_cols) throw size_mismatch();
        std::vector<T> res(n_rows, T());
        for (size_t i = 0; i < n_rows; ++i) {
            T acc = T();
            for (size_t k = indptr_[i]; k < indptr_[i + 1]; ++k) {
                acc = acc + data_[k] * vec[indices_[k]];
            }
            res[i] = acc;
        }
        return res;
    }

    // Row slicing
    CSRMatrix getRowSlice(size_t l, size_t r) const {
        if (l > r || r > n_rows) throw invalid_index();
        size_t new_n = r - l;
        CSRMatrix<T> sub(new_n, n_cols);
        sub.indptr_.assign(new_n + 1, 0);
        size_t total = 0;
        for (size_t i = 0; i < new_n; ++i) {
            size_t s = indptr_[l + i];
            size_t e = indptr_[l + i + 1];
            for (size_t k = s; k < e; ++k) {
                sub.indices_.push_back(indices_[k]);
                sub.data_.push_back(data_[k]);
                ++total;
            }
            sub.indptr_[i + 1] = total;
        }
        return sub;
    }
};

} // namespace sjtu

#endif // CSR_MATRIX_HPP

