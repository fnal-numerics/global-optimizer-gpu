#include <iostream>
#include <fstream>
#include <random>

// Function to generate random numbers and save them to a file
void generateAndSaveData(const std::string& filename, size_t totalNumbers, size_t batchSize) {
    std::ofstream outFile(filename, std::ios::binary);
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    std::vector<double> buffer;
    buffer.reserve(batchSize);

    while (totalNumbers > 0) {
        size_t currentBatchSize = std::min(batchSize, totalNumbers);
        buffer.clear();

        for (size_t i = 0; i < currentBatchSize; ++i) {
            buffer.push_back(dist(rng));
        }

        outFile.write(reinterpret_cast<char*>(buffer.data()), currentBatchSize * sizeof(double));
        totalNumbers -= currentBatchSize;
    }

    outFile.close();
}


double* readDataFromFile(const char* filename, int N) {
    std::ifstream inFile(filename, std::ios::binary);
    double* hostPoints = new double[N];

    if (!inFile.is_open()) {
        std::cerr << "Failed to open the file for reading.\n";
        return nullptr;
    }

    inFile.read(reinterpret_cast<char*>(hostPoints), N * sizeof(double));
    inFile.close();
    return hostPoints;
}

int main() {
    const size_t totalNumbers = static_cast<size_t>(1024) * 128;// * 1024 * 1024; // 1024^4
    const size_t batchSize = 1024;//*1024; // Process in batches of 1024^2
    std::cout << "Generating and saving data..." << std::endl;
    generateAndSaveData("trillion_random_numbers.bin", totalNumbers, batchSize);
    int N = 100;

    std::cout << "Reading data back from file..." << std::endl;
    double* dataReadBack = readDataFromFile(filename, N);
    if (dataReadBack != nullptr) {
        std::cout << "First 10 numbers read from file:" << std::endl;
        for (int i = 0; i < 10 && i < N; ++i) {
            std::cout << dataReadBack[i] << " ";
        }
        std::cout << std::endl;

        delete[] dataReadBack;
    } else {
        std::cerr << "Error reading data from file." << std::endl;
        return 1;
    }

    return 0;
}

