#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <utility>

std::pair<std::vector<std::vector<double>>, std::vector<double>>
generateSyntheticData(size_t numSamples, size_t numFeatures, unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> featureDist(-1.0, 1.0);
    std::uniform_real_distribution<double> noiseDist(-1.0, 1.0);

    std::vector<std::vector<double>> X(numSamples, std::vector<double>(numFeatures));
    std::vector<double> y(numSamples);


    std::vector<double> trueWeights(numFeatures);
    for (double &w : trueWeights)
    {
        w = featureDist(rng);
    }

    for (size_t i = 0; i < numSamples; ++i)
    {
        double sum = 0.0;

        for (size_t j = 0; j < numFeatures; ++j)
        {
            X[i][j] = featureDist(rng);
            sum += X[i][j] * trueWeights[j];
        }

        sum += noiseDist(rng); 
        y[i] = sum > 0 ? 1.0 : 0.0;
    }
    size_t ones = std::count(y.begin(), y.end(), 1.0);
    size_t zeros = y.size() - ones;
    std::cout << "1s: " << ones << ", 0s: " << zeros << "\n";

    return {X, y};
}
