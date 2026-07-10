#include "turbo/loader/gguf_loader.hpp"
#include "turbo/tensor/tensor.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_gguf_file>\n";
        return 1;
    }

    try {
        turbo::GGUFLoader loader(argv[1]);
        loader.parse();

        std::cout << "\n--- Fetching Target Weight Vector via Zero-Copy ---\n";
        
        // Query the tensor we declared in the Python dummy generator
        std::string target_weight = "blk.0.attn_q.weight";
        Tensor q_weight = loader.get_tensor(target_weight);
        
        std::cout << "Successfully bound weight: " << target_weight << "\n";
        std::cout << "Data Address in RAM: " << q_weight.data_ptr() << "\n";
        std::cout << "Dimensions Tracked: " << q_weight.get_shape().size() << "\n";
        std::cout << "Format Code (ggml_type): " << static_cast<int>(q_weight.dtype()) << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Execution crash: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
