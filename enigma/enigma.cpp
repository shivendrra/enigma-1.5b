#include <torch/torch.h>
#include <iostream>
#include <vector>

// Define device
torch::Device device(torch::kCUDA);

// Define constants
const int batch_size = 8;
const int block_size = 32;
const int max_iters = 1000;
const int eval_interval = 50;
const int eval_iters = 5;
const int d_model = 256;
const int n_layer = 16;
const int n_head = 12;
const float dropout = 0.2;
const float norm_eps = 1e-5;
const int vocab_size = 5;

// sample data
torch::Tensor train_data = torch::rand({1000, block_size});
torch::Tensor val_data = torch::rand({500, block_size});

// Data loading function
std::pair<torch::Tensor, torch::Tensor> get_batch(const std::string& split) {
    torch::Tensor data = (split == "train") ? train_data : val_data;
    torch::Tensor ix = torch::randint(data.size(0) - block_size, {batch_size});
    torch::Tensor x = torch::empty({batch_size, block_size});
    torch::Tensor y = torch::empty({batch_size, block_size});
    for (int i = 0; i < batch_size; ++i) {
        x[i] = data.index({ix[i], ix[i] + block_size});
        y[i] = data.index({ix[i] + 1, ix[i] + block_size + 1});
    }
    return std::make_pair(x.to(device), y.to(device));
}

// Custom classes and functions
class SWiGLU : public torch::nn::Module {
public:
    SWiGLU() {}

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor sigmoid_output = torch::sigmoid(x);
        torch::Tensor relu_output = torch::relu(x);
        torch::Tensor out = sigmoid_output * relu_output + (1 - sigmoid_output) * x;
        return out;
    }
};

class UnMaskedHeadImpl : public torch::nn::Module {
public:
    UnMaskedHeadImpl(int d_model, int head_size, float dropout)
        : key(register_module("key", torch::nn::Linear(d_model, head_size))),
          query(register_module("query", torch::nn::Linear(d_model, head_size))),
          value(register_module("value", torch::nn::Linear(d_model, head_size))),
          dropout(torch::nn::Dropout(dropout)) {
        register_module("dropout", dropout);
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor key_out = key->forward(x);
        torch::Tensor query_out = query->forward(x);
        
        torch::Tensor weights = query_out.matmul(key_out.transpose(-2, -1)) * std::sqrt(key_out.size(-1));
        weights = torch::softmax(weights, -1);
        weights = dropout(weights);

        torch::Tensor value_out = value->forward(x);
        torch::Tensor out = weights.matmul(value_out);
        return out;
    }

private:
    torch::nn::Linear key, query, value;
    torch::nn::Dropout dropout;
};

TORCH_MODULE(UnMaskedHead);

class MaskedHeadImpl : public torch::nn::Module {
public:
    MaskedHeadImpl(int head_size, float dropout, int d_model)
        : key(register_module("key", torch::nn::Linear(d_model, head_size))),
          query(register_module("query", torch::nn::Linear(d_model, head_size))),
          value(register_module("value", torch::nn::Linear(d_model, head_size))),
          dropout(torch::nn::Dropout(dropout)) {
        register_buffer("tril", torch::tril(torch::ones(block_size, block_size)));
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor key_out = key->forward(x);
        torch::Tensor query_out = query->forward(x);
        
        torch::Tensor weights = query_out.matmul(key_out.transpose(-2, -1)) * std::sqrt(key_out.size(-1));
        weights = weights.masked_fill(tril[:x.size(1), :x.size(1)] == 0, std::numeric_limits<float>::lowest());
        weights = torch::softmax(weights, -1);
        weights = dropout(weights);

        torch::Tensor value_out = value->forward(x);
        torch::Tensor out = weights.matmul(value_out);
        return out;
    }

private:
    torch::nn::Linear key, query, value;
    torch::nn::Dropout dropout;
    torch::Tensor tril;
};

TORCH_MODULE(MaskedHead);

class MultiUnMaskedImpl : public torch::nn::Module {
public:
    MultiUnMaskedImpl(int d_model, int n_head, float dropout)
        : proj(register_module("proj", torch::nn::Linear(n_head * (d_model / n_head), d_model))),
          dropout(torch::nn::Dropout(dropout)) {
        for (int i = 0; i < n_head; ++i) {
            heads.push_back(register_module("head" + std::to_string(i), UnMaskedHead(d_model, d_model / n_head, dropout)));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        std::vector<torch::Tensor> head_outputs;
        for (auto& head : heads) {
            head_outputs.push_back(head->forward(x));
        }
        torch::Tensor out = torch::cat(head_outputs, -1);
        out = dropout(out);
        out = proj(out);
        return out;
    }

private:
    torch::nn::Linear proj;
    torch::nn::Dropout dropout;
    std::vector<UnMaskedHead> heads;
};

TORCH_MODULE(MultiUnMasked);

class MultiMaskedImpl : public torch::nn::Module {
public:
    MultiMaskedImpl(int d_model, int n_head, float dropout)
        : proj(register_module("proj", torch::nn::Linear(n_head * (d_model / n_head), d_model))),
          dropout(torch::nn::Dropout(dropout)) {
        for (int i = 0; i < n_head; ++i) {
            heads.push_back(register_module("head" + std::to_string(i), MaskedHead(d_model, d_model / n_head, dropout)));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        std::vector<torch::Tensor> head_outputs;
        for (auto& head : heads) {
            head_outputs.push_back(head->forward(x));
        }
        torch::Tensor out = torch::cat(head_outputs, -1);
        out = dropout(out);
        out = proj(out);
        return out;
    }

private:
    torch::nn::Linear proj;
    torch::nn::Dropout dropout;
    std::vector<MaskedHead> heads;
};

TORCH_MODULE(MultiMasked);

class FeedForwardImpl : public torch::nn::Module {
public:
    FeedForwardImpl(int d_model, float dropout)
        : net(register_module("net", torch::nn::Sequential(
            torch::nn::Linear(d_model, 4 * d_model),
            torch::nn::GELU(),
            torch::nn::Linear(4 * d_model, d_model),
            torch::nn::Dropout(dropout)
        ))) {}

    torch::Tensor forward(torch::Tensor x) {
        return net->forward(x);
    }

private:
    torch::nn::Sequential net;
};

TORCH_MODULE(FeedForward);

class BlockImpl : public torch::nn::Module {
public:
    BlockImpl(int d_model, int n_head, float norm_eps, float dropout)
        : sa_masked(MultiMasked(d_model, n_head, dropout)),
          sa_unmasked(MultiUnMasked(d_model, n_head, dropout)),
          ffwd(FeedForward(d_model, dropout)),
          norm1(torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}).eps(norm_eps))),
          norm2(torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}).eps(norm_eps))) {}

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor x2 = x + sa_unmasked->forward(norm1->forward(x));
        x = x2 + ffwd->forward(norm2->forward(x2));

        x2 = x + sa_masked->forward(norm1->forward(x));
        x = x2 + ffwd->forward(norm2->forward(x2));

        return x;
    }

private:
    MultiMasked sa_masked;
    MultiUnMasked sa_unmasked;
    FeedForward ffwd;
    torch::nn::LayerNorm norm1, norm2;
};

TORCH_MODULE(Block);

class EnigmaImpl : public torch::nn::Module {
public:
    EnigmaImpl(int vocab_size, int block_size, int d_model, int n_layer, int n_head, float dropout, float norm_eps)
        : toked_model(register_module("toked_model", torch::nn::Embedding(vocab_size, d_model))),
          pos_encod(register_module("pos_encod", torch::nn::Embedding(block_size, d_model))),
          norm_final(torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}).eps(norm_eps))),
          linear_final(register_module("linear_final", torch::nn::Linear(d_model, vocab_size))) {
        for (int i = 0; i < n_layer; ++i) {
            block_layers.push_back(register_module("block" + std::to_string(i), Block(d_model, n_head, norm_eps, dropout)));
        }
        register_buffer("block_size", torch::tensor(block_size));
        _init_weights(this);
    }

    void _init_weights(torch::nn::Module* module) {
        auto parameters = module->named_parameters();
        for (auto& param : parameters) {
            if (param.key().find("weight") != std::string::npos) {
                torch::nn::init::normal_(param.value(), 0.0, 0.02);
            } else if (param.key().find("bias") != std::string::npos) {
                torch::nn::init::zeros_(param.value());
            }
        }
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor idx, torch::Tensor targets=torch::Tensor()) {
        torch::Tensor toked_model_out = toked_model->forward(idx);
        torch::Tensor pos_encod_out = pos_encod->forward(torch::arange(idx.size(1)));
        torch::Tensor x = toked_model_out + pos_encod_out;

        for (auto& block : block_layers) {
            x = block->forward(x);
        }

        torch::Tensor logits = linear_final->forward(norm_final->forward(x));

        if (!targets.numel()) {
            return {logits, torch::Tensor()};
        } else {
            logits = logits.view({-1, logits.size(-1)});
            targets = targets.view({-1});
            torch::Tensor loss = torch::nn::functional::cross_entropy(logits, targets);
            return {logits, loss};
        }
    }

    std::vector<std::vector<std::pair<torch::Tensor, float>>> complex_generate(torch::Tensor idx, int max_new_tokens, float temperature=1.0, int top_k=3, int beam_width=5) {
        std::vector<std::vector<std::pair<torch::Tensor, float>>> completed_beams;
        torch::Tensor current_idx = idx.clone();
        std::vector<std::pair<torch::Tensor, float>> beam = {std::make_pair(current_idx, 0.0)};

        for (int i = 0; i < max_new_tokens; ++i) {
            std::vector<std::pair<torch::Tensor, float>> new_beam;

            for (auto& beam_item : beam) {
                torch::Tensor& current_idx = beam_item.first;
                torch::Tensor logits, loss;
                std::tie(logits, loss) = forward(current_idx);
                logits = logits.index({torch::indexing::Slice(), -1}); // Get last token predictions

                // Apply softmax and temperature
                torch::Tensor probs = torch::nn::functional::softmax(logits / temperature, -1);
                
                // Top-k sampling
                if (top_k > 0) {
                    probs = top_k_filtering(probs, top_k);
                }

                // Sample from the distribution
                torch::Tensor sampled_idx = torch::multinomial(probs, beam_width, true);

                for (int j = 0; j < beam_width; ++j) {
                    torch::Tensor new_idx = torch::cat({current_idx, sampled_idx.index({torch::indexing::Slice(), j})}, 1);
                    torch::Tensor new_log_prob = beam_item.second + torch::log(probs.index({torch::indexing::Slice(), sampled_idx.index({torch::indexing::Slice(), j})}));
                    new_beam.push_back(std::make_pair(new_idx, new_log_prob.item()));
                }
            }

            // Sort new beam by log probabilities
            std::sort(new_beam.begin(), new_beam.end(), [](const std::pair<torch::Tensor, float>& a, const std::pair<torch::Tensor, float>& b) {
                return a.second > b.second;
            });

            // Only keep top beams
            beam = std::vector<std::pair<torch::Tensor, float>>(new_beam.begin(), new_beam.begin() + beam_width);
        }

        completed_beams.push_back(beam);
        return completed_beams;
    }

    std::vector<std::vector<std::pair<torch::Tensor, float>>> top_k_filtering(torch::Tensor logits, int top_k) {
        torch::Tensor top_values, top_indices;
        std::tie(top_values, top_indices) = torch::topk(logits, top_k, -1);

        torch::Tensor min_value = torch::index_select(top_values, -1, torch::tensor({top_k-1}));
        torch::Tensor filtered_logits = torch::where(logits < min_value, torch::full_like(logits, -std::numeric_limits<float>::infinity()), logits);
        return filtered_logits;
    }

private:
    torch::nn::Embedding toked_model, pos_encod;
    std::vector<Block> block_layers;
    torch::nn::LayerNorm norm_final;
    torch::nn::Linear linear_final;
    int block_size;
};

TORCH_MODULE(Enigma);

int main() {
    // Set seed
    torch::manual_seed(1400);

    // Create model
    Enigma model(vocab_size, block_size, d_model, n_layer, n_head, dropout, norm_eps);
    model->to(device);

    // Define optimizer
    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(learning_rate));

    // Training loop
    std::vector<float> train_losses, val_losses;
    for (int iter = 0; iter < max_iters; ++iter) {
        if (iter % eval_interval == 0 || iter == max_iters - 1) {
            // Evaluate and print losses
            auto losses = estimate_loss();
            std::cout << "step " << iter << ": train loss " << losses["train"] << ", val loss " << losses["val"] << std::endl;
            
            // Save losses for plotting
            train_losses.push_back(losses["train"]);
            val_losses.push_back(losses["val"]);
        }

        // Get batch, forward pass, loss calculation, backward pass, optimizer step
        auto [xb, yb] = get_batch("train");
        torch::Tensor logits, loss;
        std::tie(logits, loss) = model->forward(xb, yb);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    return 0;
}
