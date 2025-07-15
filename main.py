from cache import RotatingBufferCache
import logging
import torch
import fire
from typing import List
from pathlib import Path

from model import Transformer
from tokenizer import Tokenizer


def top_p_sampling(probabilities: torch.Tensor, threshold_p: float):
    assert 0.0 <= threshold_p <= 1.0, "Top-p value must be between 0 and 1."

    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff_mask = cumulative_probs - sorted_probs > threshold_p
    sorted_probs[cutoff_mask] = 0.0

    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    sampled_token = torch.multinomial(sorted_probs, 1)

    return torch.gather(sorted_indices, -1, sampled_token)



def token_sampling(logits_tensor: torch.Tensor, temp: float, nucleus_p: float):
    if temp > 0.0:
        scaled_probs = torch.softmax(logits_tensor / temp, dim=-1)
        token = top_p_sampling(scaled_probs, nucleus_p)
    else:
        token = torch.argmax(logits_tensor, dim=-1, keepdim=True)

    return token.view(-1)



@torch.inference_mode()
def generate_text(prompts: List[str], model: Transformer, tokenizer: Tokenizer, *, max_new_tokens: int, temperature: float, chunk_step: int = None):
    model.eval()

    batch_size = len(prompts)
    vocab_size = model.args.vocab_size
    
    # Tokenization
    encoded_inputs = [tokenizer.encode(text, bos=True) for text in prompts]
    input_lengths = [len(seq) for seq in encoded_inputs]

    # Setup cache (rotating buffer)
    total_cache_size = max(input_lengths) + max_new_tokens
    
    if model.args.sliding_window is not None and total_cache_size > model.args.sliding_window:
        total_cache_size = model.args.sliding_window

    buffer_cache = RotatingBufferCache(
        model.n_local_layers,
        model.args.max_batch_size,
        total_cache_size,
        model.args.n_kv_heads,
        model.args.head_dim,
    )

    buffer_cache.to(device=model.device, dtype=model.dtype)
    buffer_cache.reset()
    
    # Tracking
    token_logprobs = [[] for _ in range(batch_size)]
    previous_logits = None

    # Determine chunk processing size
    max_input_len = max(input_lengths)
    if chunk_step is None:
        chunk_step = max_input_len

    # Chunked prompt encoding
    for start_idx in range(0, max_input_len, chunk_step):
        current_chunks = [seq[start_idx : start_idx + chunk_step] for seq in encoded_inputs]
        assert all(len(chunk) > 0 for chunk in current_chunks)

        input_tensor = torch.tensor(
            sum(current_chunks, []),
            device=model.device,
            dtype=torch.long
        )

        pre_logits = model.forward(
            input_tensor,
            seqlens=[len(seq) for seq in current_chunks],
            cache=buffer_cache
        )

        logits_tensor = torch.log_softmax(pre_logits, dim=-1)

        if previous_logits is not None:
            logits_last = torch.log_softmax(previous_logits, dim=-1)
            for seq_idx in range(batch_size):
                first_token_idx = current_chunks[seq_idx][0]
                token_logprobs[seq_idx].append(logits_last[seq_idx, first_token_idx].item())

        # Accumulate logprobs for current chunk
        offset_idx = 0
        for seq_idx, sequence in enumerate(current_chunks):
            log_probs_for_seq = [
                logits_tensor[offset_idx + tok_idx, sequence[tok_idx + 1]].item()
                for tok_idx in range(len(sequence) - 1)
            ]
            token_logprobs[seq_idx].extend(log_probs_for_seq)
            offset_idx += len(sequence)

        # Update last logits using final tokens of each sequence
        token_positions = torch.tensor([len(seq) for seq in current_chunks], device=pre_logits.device).cumsum(dim=0) - 1
        previous_logits = pre_logits.index_select(0, token_positions)
        assert previous_logits.shape == (batch_size, vocab_size)

    # Token generation loop
    next_tokens_collected = []
    assert previous_logits is not None

    for _ in range(max_new_tokens):
        sampled_token = token_sampling(previous_logits, temp=temperature, nucleus_p=0.8)

        last_logits = torch.log_softmax(previous_logits, dim=-1)
        for seq_idx in range(batch_size):
            token_logprobs[seq_idx].append(last_logits[seq_idx, sampled_token[seq_idx]].item())

        next_tokens_collected.append(sampled_token[:, None])
        previous_logits = model.forward(sampled_token, seqlens=[1] * batch_size, cache=buffer_cache)

        assert previous_logits.shape == (batch_size, vocab_size)

    # Decode the full output
    final_texts = []
    if next_tokens_collected:
        all_generated_tokens = torch.cat(next_tokens_collected, dim=1)
        for seq_idx, original_tokens in enumerate(encoded_inputs):
            full_sequence = original_tokens + all_generated_tokens[seq_idx].tolist()
            final_texts.append(tokenizer.decode(full_sequence))

    return final_texts, token_logprobs


def chat_session(model_directory: str, max_generated_tokens: int = 35, temp: float = 0.7, use_instruction: bool = False):
    tokenizer = Tokenizer(str(Path(model_directory) / "tokenizer.model"))
    model = Transformer.from_folder(Path(model_directory), max_batch_size=3)

    while True:
        user_input = input("Enter your prompt: ")
        if use_instruction:
            user_input = f"[INST] {user_input} [/INST]"

        responses, _ = generate_text(
            [user_input],
            model,
            tokenizer,
            max_new_tokens=max_generated_tokens,
            temperature=temp,
        )
        print(responses[0])
        print("-----------")


def run_demo(
    model_directory: str,
    max_generated_tokens: int = 35,
    temp: float = 0.0,
    pipeline_ranks: int = 1
):
    if pipeline_ranks > 1:
        torch.distributed.init_process_group()
        torch.cuda.set_device(torch.distributed.get_rank())
        display_output = torch.distributed.get_rank() == 0
    else:
        display_output = True

    tokenizer = Tokenizer(str(Path(model_directory) / "tokenizer.model"))
    model = Transformer.from_folder(
        Path(model_directory),
        max_batch_size=3,
        num_pipeline_ranks=pipeline_ranks
    )

    prompts = [
        "I'm working on an AI project. Can you recommend some datasets?",
        "Tell me a quick joke.",
        "Explain why Mistral AI models are efficient."
    ]

    generated_texts, log_probs = generate_text(
        prompts,
        model,
        tokenizer,
        max_new_tokens=max_generated_tokens,
        temperature=temp
    )

    if display_output:
        for result, logs in zip(generated_texts, log_probs):
            print(result)
            logging.debug("Logprobs: %s", logs)
            print("-----------")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire({
        "chat": chat_session,
        "demo": run_demo,
    })
