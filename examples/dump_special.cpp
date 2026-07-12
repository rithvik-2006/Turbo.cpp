#include <iostream>
#include "turbo/loader/gguf_loader.hpp"
int main(int argc, char** argv) {
    turbo::GGUFLoader loader(argv[1]);
    loader.parse();
    // we need to add a method to get metadata array or just hack it
    return 0;
}
