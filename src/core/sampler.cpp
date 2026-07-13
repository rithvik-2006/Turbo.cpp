#include "../../include/turbo/core/sampler.hpp"
#include "../../include/turbo/tensor/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace turbo {

int greedy_sample(const float* last_token_logits, size_t vocab_size) {
  float max_val = -std::numeric_limits<float>::infinity();
  int max_index = 0;

  for (size_t i = 0; i < vocab_size; ++i) {
    float current_val = last_token_logits[i];
    if (current_val > max_val) {
      max_val = current_val;
      max_index = static_cast<int>(i);
    }
  }

  return max_index;
}

int sample_top_p(const float* last_token_logits, size_t vocab_size, float temperature, float top_p) {
    // 1. Apply Temperature
    std::vector<float> scaled_logits(vocab_size);
    float max_logit = -std::numeric_limits<float>::infinity();
    
    for (size_t i = 0; i < vocab_size; ++i) {
        scaled_logits[i] = last_token_logits[i] / temperature;
        if (scaled_logits[i] > max_logit) {
            max_logit = scaled_logits[i];
        }
    }

    // 2. Numerically Stable Softmax
    std::vector<float> probs(vocab_size);
    float sum_probs = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(scaled_logits[i] - max_logit);
        sum_probs += probs[i];
    }
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] /= sum_probs;
    }

    // 3. Pair probabilities with Token IDs for sorting
    std::vector<std::pair<float, int>> prob_id_pairs(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        prob_id_pairs[i] = {probs[i], static_cast<int>(i)};
    }

    // Sort descending by probability
    std::sort(prob_id_pairs.begin(), prob_id_pairs.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // 4. Apply Top-P (Nucleus) Filtering
    float cumulative_prob = 0.0f;
    std::vector<std::pair<float, int>> filtered_pairs;
    
    for (const auto& pair : prob_id_pairs) {
        filtered_pairs.push_back(pair);
        cumulative_prob += pair.first;
        if (cumulative_prob >= top_p) {
            break; 
        }
    }

    // 5. Re-normalize the filtered probabilities
    std::vector<float> final_probs(filtered_pairs.size());
    for (size_t i = 0; i < filtered_pairs.size(); ++i) {
        final_probs[i] = filtered_pairs[i].first / cumulative_prob;
    }

    // 6. Sample randomly from the remaining distribution
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<int> distribution(final_probs.begin(), final_probs.end());
    
    int selected_index = distribution(gen);
    return filtered_pairs[selected_index].second;
}

} // namespace turbo