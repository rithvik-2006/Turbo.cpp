sed -i 's/"\.ffn_gate\.weight"/"\.ffn_up\.weight"/g' src/nn/transforrmer_block.cpp
sed -i 's/"\.ffn_up\.weight"/"\.ffn_gate_TEMP\.weight"/g' src/nn/transforrmer_block.cpp
sed -i 's/"\.ffn_gate_TEMP\.weight"/"\.ffn_gate\.weight"/g' src/nn/transforrmer_block.cpp
