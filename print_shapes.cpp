#include <iostream>
#include "turbo/loader/gguf_loader.hpp"
#include "turbo/tensor/tensor.hpp"
using namespace turbo;
int main(int argc, char** argv) {
    try {
        GGUFLoader loader(argv[1]);
        loader.parse();
        auto wk = loader.get_tensor("blk.0.attn_k.weight");
        std::cout << "blk.0.attn_k.weight shape: [";
        for(auto s : wk.get_shape()) std::cout << s << ", ";
        std::cout << "]" << std::endl;
    } catch (std::exception& e) { std::cerr << e.what() << std::endl; }
    return 0;
}
