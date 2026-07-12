#include <iostream>
#include "turbo/loader/gguf_loader.hpp"
int main(int argc, char** argv) {
    turbo::GGUFLoader loader(argv[1]);
    loader.parse();
    const auto& vocab = loader.vocab_tokens();
    std::vector<int> ids = {1, 29966, 29989, 5205, 29989, 29958, 13, 19423, 1792, 465, 22137, 20255};
    for (int id : ids) {
        if (id < vocab.size()) std::cout << id << " : " << vocab[id] << "\n";
    }
    return 0;
}
