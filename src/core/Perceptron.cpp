#include "core/Perceptron.hpp"
#include <random>

Perceptron::Perceptron(size_t inputSize)
    : neuron(0.0, [&]
             {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        std::vector<double> initWeights(inputSize);
        for (auto& w : initWeights) w = dist(rng);
        return initWeights; }()) {}

void Perceptron::train(
    const std::vector<std::vector<double>> &X_train,
    const std::vector<double> &y_train,
    size_t epochs,
    double learningRate)
{
    for (size_t epo = 0; epo < epochs; ++epo)
    {
        std::vector<double> y_pred;

        for (size_t iter = 0; iter < X_train.size(); ++iter)
        {
            const std::vector<double> &x = X_train[iter];
            double y_true = y_train[iter];

            double prediction = neuron.Compute(x); 
            y_pred.push_back(prediction);
            neuron.UpdateWeightsCached(prediction, x, y_true, learningRate);
        }

        double loss = ComputeLoss(y_train, y_pred);
        std::cout << "Epoch [" << epo << "] : loss = " << loss << "\n";
    }
}

int Perceptron::predict(const std::vector<double> &input)
{

    double predictionResult = neuron.Compute(input);
    return predictionResult >= 0.5 ? 1 : 0;
}

double Perceptron::evaluate(const std::vector<std::vector<double>> &X_test,
                            const std::vector<double> &y_test)
{
    size_t correct = 0;

    for (size_t i = 0; i < X_test.size(); ++i)
    {
        int prediction = predict(X_test[i]);
        if (prediction == y_test[i])
        {
            ++correct;
        }
    }

    return static_cast<double>(correct) / y_test.size();
}