#pragma once

#include <vector>
#include "core/Neuron.hpp"


class Perceptron{
public:
    Perceptron(size_t inputSize);

    void train(const std::vector<std::vector<double>> &X_train,
               const std::vector<double> &y_train,
               size_t epochs,
               double learningRate);
    int predict(const std::vector<double> &input);

    double evaluate(const std::vector<std::vector<double>>& X_test,
                            const std::vector<double>& y_test);
private:
    Neuron neuron;
};