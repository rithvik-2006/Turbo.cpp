#include <iostream>
#include "turbo/loader/gguf_loader.hpp"
int main(int argc, char** argv) {
    turbo::GGUFLoader loader(argv[1]);
    loader.parse();
    return 0;
}
