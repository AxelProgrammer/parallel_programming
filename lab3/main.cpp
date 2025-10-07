#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h> // добавлено для OpenMP

const int MATRIX_SIZE = 10;
const int NUM_MATRICES = 1000;
const int MIN_VAL = -10;
const int MAX_VAL = 10;

using Matrix = std::vector<std::vector<int>>;

// Функция генерации случайной матрицы
Matrix generate_matrix(std::mt19937 &rng, std::uniform_int_distribution<int> &dist) {
    Matrix mat(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
    for (int i = 0; i < MATRIX_SIZE; ++i)
        for (int j = 0; j < MATRIX_SIZE; ++j)
            mat[i][j] = dist(rng);
    return mat;
}

// Произведение Адамара двух матриц
Matrix hadamard_product(const Matrix &A, const Matrix &B) {
    Matrix result(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
    for (int i = 0; i < MATRIX_SIZE; ++i)
        for (int j = 0; j < MATRIX_SIZE; ++j)
            result[i][j] = A[i][j] * B[i][j];
    return result;
}

int main() {
    // 1. Генерация исходных матриц
    std::vector<Matrix> matrices;
    matrices.reserve(NUM_MATRICES);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(MIN_VAL, MAX_VAL);

    for (int i = 0; i < NUM_MATRICES; ++i)
        matrices.push_back(generate_matrix(rng, dist));

    // 2. Генерация всех уникальных пар матриц
    std::vector<std::pair<int,int>> pairs;
    for (int i = 0; i < NUM_MATRICES; ++i)
        for (int j = i + 1; j < NUM_MATRICES; ++j)
            pairs.emplace_back(i,j);

    int total_products = pairs.size();
    std::vector<Matrix> results(total_products);

    // Количество потоков для эксперимента
    std::vector<int> thread_counts = {1, 2, 4, 8};

    // 1: #pragma omp parallel без #pragma omp for
    std::cout << "=== Parallel without omp for ===\n";
    for (int num_threads : thread_counts) {
        omp_set_num_threads(num_threads); // устанавливаем количество потоков

        auto start_time = std::chrono::high_resolution_clock::now();

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();   // ID потока
            int threads_count = omp_get_num_threads(); // число потоков

            // делим пары между потоками вручную
            int chunk_size = (total_products + threads_count - 1) / threads_count;
            int start_idx = thread_id * chunk_size;
            int end_idx = std::min(start_idx + chunk_size, total_products);

            for (int idx = start_idx; idx < end_idx; ++idx) {
                auto [i,j] = pairs[idx];
                results[idx] = hadamard_product(matrices[i], matrices[j]);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "Threads: " << num_threads
                  << ", Time: " << elapsed.count() << " s\n";
    }

    // 2: #pragma omp parallel + #pragma omp for
    std::cout << "\n=== Параллельно с omp для ===\n";
    for (int num_threads : thread_counts) {
        omp_set_num_threads(num_threads);

        auto start_time = std::chrono::high_resolution_clock::now();

        #pragma omp parallel
        {
            #pragma omp for // OpenMP сам распределяет итерации между потоками
            for (int idx = 0; idx < total_products; ++idx) {
                auto [i,j] = pairs[idx];
                results[idx] = hadamard_product(matrices[i], matrices[j]);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "Threads: " << num_threads
                  << ", Time: " << elapsed.count() << " s\n";
    }

    return 0;
}
