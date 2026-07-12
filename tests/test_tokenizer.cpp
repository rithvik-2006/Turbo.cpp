#include "turbo/core/tokenizer.hpp"
#include "turbo/loader/gguf_loader.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <fstream>

TEST(TokenizerTest, BasicBPEEncodeDecode) {
    using namespace turbo;

    // Vocab structure matching our step-by-step example:
    // 0: <unk>, 1: <s>, 2: </s>
    // Characters: h, e, l, o, w, r, d, [space], !
    // Merges: he(0.5), ll(0.2), lo(0.6), hel(0.8), wo(0.5), wor(0.6), worl(0.7), world(0.9), hello(1.0)
    std::vector<std::string> vocab = {
        "<unk>", "<s>", "</s>", "h", "e", "l", "o", "w", "r", "d", "\xe2\x96\x81", "!", "he", "ll", "lo", "hel", "wo", "wor", "worl", "world", "hello"
    };
    std::vector<float> scores = {
        0.0f, 0.0f, 0.0f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.5f, 0.2f, 0.6f, 0.8f, 0.5f, 0.6f, 0.7f, 0.9f, 1.0f
    };

    Tokenizer tokenizer(vocab, scores, 1, 2, 0);

    // 1. Test encoding "hello" without BOS
    std::vector<int> tokens_hello = tokenizer.encode("hello", false);
    // Expected: ["hello"] which maps to [20]
    ASSERT_EQ(tokens_hello.size(), 1);
    EXPECT_EQ(tokens_hello[0], 20);

    // 2. Test encoding "hello world!" with BOS
    std::vector<int> tokens_world = tokenizer.encode("hello world!", true);
    // Expected: [<s>, hello, " ", world, !] -> [1, 20, 10, 19, 11]
    std::vector<int> expected_world = {1, 20, 10, 19, 11};
    ASSERT_EQ(tokens_world.size(), expected_world.size());
    for (size_t i = 0; i < expected_world.size(); ++i) {
        EXPECT_EQ(tokens_world[i], expected_world[i]);
    }

    // 3. Test decoding
    std::string decoded_world = tokenizer.decode(tokens_world);
    EXPECT_EQ(decoded_world, "hello world!");

    // 4. Test unknown token handling
    // character 'x' is not in vocabulary, should map to <unk> (0)
    std::vector<int> tokens_unk = tokenizer.encode("x", false);
    ASSERT_EQ(tokens_unk.size(), 1);
    EXPECT_EQ(tokens_unk[0], 0);
}

TEST(TokenizerTest, GGUFIntegration) {
    using namespace turbo;

    try {
        std::string model_path = "dummy_model.gguf";
        std::ifstream f(model_path.c_str());
        if (!f.good()) {
            model_path = "../../dummy_model.gguf";
        }
        GGUFLoader loader(model_path);
        loader.parse();

        const auto& vocab = loader.vocab_tokens();
        const auto& scores = loader.vocab_scores();

        // Check if vocab was extracted successfully
        ASSERT_FALSE(vocab.empty());
        ASSERT_FALSE(scores.empty());
        ASSERT_EQ(vocab.size(), scores.size());

        EXPECT_EQ(vocab[0], "<unk>");
        EXPECT_EQ(vocab[1], "<s>");
        EXPECT_EQ(vocab[2], "</s>");
        EXPECT_EQ(vocab[20], "hello");

        // Construct Tokenizer using GGUF loader vocab
        Tokenizer tokenizer(vocab, scores, 1, 2, 0);

        std::vector<int> tokens = tokenizer.encode("hello world!", true);
        std::vector<int> expected = {1, 20, 10, 19, 11};
        ASSERT_EQ(tokens.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(tokens[i], expected[i]);
        }

        std::string decoded = tokenizer.decode(tokens);
        EXPECT_EQ(decoded, "hello world!");

    } catch (const std::exception& e) {
        FAIL() << "Failed to parse GGUF or tokenization crashed: " << e.what();
    }
}

TEST(TokenizerTest, BPE_HelloWorld_Merge) {
    using namespace turbo;
    
    // 1. Create a controlled mock vocabulary based on LLaMA rules
    // Note the use of the SentencePiece meta-space ' ' (U+2581)
    std::vector<std::string> vocab = {
        "<unk>", "<s>", "</s>", 
        "H", "e", "l", "o", "\xe2\x96\x81", "W", "r", "d",  // Individual chars
        "el", "lo", "Hel", "Hello",              // First word merges
        "\xe2\x96\x81W", "or", "ld", "\xe2\x96\x81World",               // Second word merges
        "\xe2\x96\x81Wor", "\xe2\x96\x81Worl" // Intermediate merges
    };
    
    // Higher scores = higher merge priority in our BPE logic
    std::vector<float> scores = {
        0.0f, 0.0f, 0.0f, 
        -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, // chars
        -5.0f, -5.0f, -2.0f, 
        10.0f,  // "Hello" (High priority merge)
        -3.0f, -3.0f, -3.0f, 
        10.0f,   // " World" (High priority merge)
        -1.0f, -1.0f
    };

    // 2. Initialize the tokenizer (BOS=1, EOS=2, UNK=0)
    Tokenizer tokenizer(vocab, scores, 1, 2, 0);

    // 3. (SentencePiece Pre-Processing is now internal to Tokenizer::encode)
    std::string prompt = "Hello World";

    // 4. Run the encoder
    std::vector<int> tokens = tokenizer.encode(prompt, true);

    // 5. Verify the Token IDs
    // Expected output:
    // [1]  -> "<s>" (BOS)
    // [14] -> "Hello"
    // [18] -> " World"
    std::vector<int> expected_ids = {1, 14, 18};

    EXPECT_EQ(tokens.size(), expected_ids.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i], expected_ids[i]);
    }
}
