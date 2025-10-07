#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>

// Размер матрицы и количество матриц
const int MATRIX_SIZE = 10;
const int NUM_MATRICES = 1000;
const int MIN_VAL = -10;  // минимальное значение элементов
const int MAX_VAL = 10;   // максимальное значение элементов

using Matrix = std::vector<std::vector<int>>;

// Функция генерации случайной матрицы 10x10
Matrix generate_matrix(std::mt19937 &rng, std::uniform_int_distribution<int> &dist) {
    Matrix mat(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
    for (int i = 0; i < MATRIX_SIZE; ++i)
        for (int j = 0; j < MATRIX_SIZE; ++j)
            mat[i][j] = dist(rng); // Заполняем случайными числами
    return mat;
}

// Функция произведения Адамара двух матриц
Matrix hadamard_product(const Matrix &A, const Matrix &B) {
    Matrix result(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
    for (int i = 0; i < MATRIX_SIZE; ++i)
        for (int j = 0; j < MATRIX_SIZE; ++j)
            result[i][j] = A[i][j] * B[i][j]; // Умножаем соответствующие элементы
    return result;
}

// Многопоточная функция для обработки диапазона пар матриц
void compute_hadamard_range(const std::vector<Matrix> &matrices,
                            const std::vector<std::pair<int,int>> &pairs,
                            int start_idx, int end_idx,
                            std::vector<Matrix> &results) {
    for (int idx = start_idx; idx < end_idx; ++idx) {
        auto [i,j] = pairs[idx];                // Получаем индексы пар матриц
        results[idx] = hadamard_product(matrices[i], matrices[j]); // Считаем произведение Адамара
    }
}

int main() {
    // 1. Генерация исходных матриц
    std::vector<Matrix> matrices;
    matrices.reserve(NUM_MATRICES);             // резервируем память для ускорения
    std::mt19937 rng(std::random_device{}());   // генератор случайных чисел
    std::uniform_int_distribution<int> dist(MIN_VAL, MAX_VAL);

    for (int i = 0; i < NUM_MATRICES; ++i) {
        matrices.push_back(generate_matrix(rng, dist));
    }

    // 2. Генерация всех уникальных пар матриц (i,j)
    std::vector<std::pair<int,int>> pairs;
    for (int i = 0; i < NUM_MATRICES; ++i)
        for (int j = i + 1; j < NUM_MATRICES; ++j)
            pairs.emplace_back(i,j);           // пары без повторений и без i==j

    int total_products = pairs.size();           // общее количество произведений
    std::vector<Matrix> results(total_products); // вектор для хранения результатов

    // 3. Запуск многопоточных вычислений
    std::vector<int> thread_counts = {1, 2, 4, 8}; // количество потоков для эксперимента

    for (int num_threads : thread_counts) {
        std::vector<std::thread> threads(num_threads);

        auto start_time = std::chrono::high_resolution_clock::now(); // старт таймера

        // Делим пары между потоками
        int chunk_size = (total_products + num_threads - 1) / num_threads; // равномерное деление

        for (int t = 0; t < num_threads; ++t) {
            int start_idx = t * chunk_size;
            int end_idx = std::min(start_idx + chunk_size, total_products);

            // Создаем поток для обработки своей части пар
            threads[t] = std::thread(compute_hadamard_range,
                                     std::cref(matrices), // передаем матрицы по ссылке
                                     std::cref(pairs),    // передаем пары по ссылке
                                     start_idx, end_idx,
                                     std::ref(results));  // результаты по ссылке
        }

        // Ждем завершения всех потоков
        for (auto &th : threads) th.join();

        auto end_time = std::chrono::high_resolution_clock::now(); // конец таймера
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "Threads: " << num_threads
                  << ", Time: " << elapsed.count() << " s\n"; // вывод времени
    }

    return 0;
}
