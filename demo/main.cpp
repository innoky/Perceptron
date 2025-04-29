#include <iostream>
#include "core/Perceptron.hpp"
#include "external/Syntezator.hpp"
#include <vector>

int main()
{
    auto [X, y] = generateSyntheticData(1'000'000, 30);

    size_t trainSize = X.size() * 0.8;

    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + trainSize);
    std::vector<std::vector<double>> X_test(X.begin() + trainSize, X.end());

    std::vector<double> y_train(y.begin(), y.begin() + trainSize);
    std::vector<double> y_test(y.begin() + trainSize, y.end());

    Perceptron perceptron(30);
    perceptron.train(X_train, y_train, 100, 0.001);

    double result = perceptron.evaluate(X_test, y_test);
    std::cout << result;
}