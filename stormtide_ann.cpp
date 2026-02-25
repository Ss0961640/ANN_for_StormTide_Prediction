#include <iostream>
#include <vector>
#include <cmath>

// 一個簡單的 3-Layer ANN（Input -> Hidden -> Output）
// 用於示範 forward 推論流程（不含訓練）

struct NeuralNetwork {
    std::vector<std::vector<double>> W1; // hidden x input
    std::vector<std::vector<double>> W2; // output x hidden
};

// Sigmoid activation
static double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// forward: input -> hidden -> output
static std::vector<double> forward(const NeuralNetwork& nn, const std::vector<double>& x) {
    // hidden layer
    std::vector<double> h(nn.W1.size(), 0.0);
    for (size_t i = 0; i < nn.W1.size(); ++i) {
        for (size_t j = 0; j < nn.W1[i].size(); ++j) {
            h[i] += x[j] * nn.W1[i][j];
        }
        h[i] = sigmoid(h[i]);
    }

    // output layer
    std::vector<double> y(nn.W2.size(), 0.0);
    for (size_t i = 0; i < nn.W2.size(); ++i) {
        for (size_t j = 0; j < nn.W2[i].size(); ++j) {
            y[i] += h[j] * nn.W2[i][j];
        }
        y[i] = sigmoid(y[i]);
    }
    return y;
}

int main() {
    NeuralNetwork nn;

    // 這裡只是 demo 權重（實際專題你會從檔案讀）
    nn.W1 = {
        {0.1, 0.2, 0.3},
        {0.4, 0.5, 0.6},
        {0.7, 0.8, 0.9}
    };

    nn.W2 = {
        {0.1, 0.2, 0.3},
        {0.4, 0.5, 0.6},
        {0.7, 0.8, 0.9}
    };

    std::vector<double> input = {1.0, 2.0, 3.0};
    std::vector<double> pred = forward(nn, input);

    std::cout << "Prediction: ";
    for (double v : pred) std::cout << v << " ";
    std::cout << "\n";
    return 0;
}