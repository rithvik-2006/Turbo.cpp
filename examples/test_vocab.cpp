#include <iostream>
#include <string>
#include "turbo/loader/gguf_loader.hpp"
int main(int argc, char** argv) {
    turbo::GGUFLoader loader(argv[1]);
    loader.parse();
    const auto& vocab = loader.vocab_tokens();
    for (size_t i = 0; i < vocab.size(); ++i) {
        if (vocab[i].find("system") != std::string::npos ||
            vocab[i].find("user") != std::string::npos ||
            vocab[i].find("assistant") != std::string::npos ||
            vocab[i].find("<") != std::string::npos) {
            std::cout << i << " : " << vocab[i] << "\n";
        }
    }
    return 0;
}
