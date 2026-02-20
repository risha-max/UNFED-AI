"""
Pipeline test — sends a prompt through the 4-node gRPC pipeline and verifies
the output matches the full model reference from verify.py.

Requires all 4 nodes to be running:
    python -m node.server --node-index 0
    python -m node.server --node-index 1
    python -m node.server --node-index 2
    python -m node.server --node-index 3

Usage:
    python test_pipeline.py
"""

import sys
import os
import uuid

import grpc
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto"))
import config
import inference_pb2
import inference_pb2_grpc


def test_pipeline():
    prompt = "The capital of France is"
    max_new_tokens = 20
    session_id = str(uuid.uuid4())

    print(f"Prompt: {prompt!r}")
    print(f"Session: {session_id}")
    print(f"Max new tokens: {max_new_tokens}")
    print()

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    print(f"Input token IDs: {input_ids}")

    # Connect to Node 0
    channel = grpc.insecure_channel(
        config.NODE_ADDRESSES[0],
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )
    stub = inference_pb2_grpc.InferenceNodeStub(channel)

    generated_tokens = list(input_ids)

    print("\nGenerating tokens:")
    for step in range(max_new_tokens):
        if step == 0:
            # Prefill: send full prompt
            request = inference_pb2.ForwardRequest(
                session_id=session_id,
                token_ids=input_ids,
                is_prefill=True,
            )
        else:
            # Decode: send only the last generated token
            request = inference_pb2.ForwardRequest(
                session_id=session_id,
                token_ids=[generated_tokens[-1]],
                is_prefill=False,
            )

        response = stub.Forward(request)

        if response.has_token:
            token_id = response.token_id
            generated_tokens.append(token_id)
            token_text = tokenizer.decode([token_id])
            print(f"  Token {step}: {token_id} ({token_text!r})")

            if response.is_eos:
                print("  [EOS reached]")
                break
        else:
            print(f"  Token {step}: No token returned (unexpected)")
            break

    full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"\nFull output: {full_text!r}")

    # Expected from verify.py reference run
    expected_tokens = [785, 6722, 315, 9625, 374, 12095, 13, 1084, 374, 279,
                       7772, 3283, 304, 4505, 323, 279, 2086, 7772, 304, 279,
                       1879, 13, 1084, 374, 1083]

    generated_part = generated_tokens[len(input_ids):]
    expected_part = expected_tokens[len(input_ids):]

    min_len = min(len(generated_part), len(expected_part))
    match = generated_part[:min_len] == expected_part[:min_len]

    if match:
        print("\nPASS: Pipeline output matches full model reference!")
    else:
        print("\nFAIL: Pipeline output differs from reference!")
        for i in range(min_len):
            g = generated_part[i]
            e = expected_part[i]
            status = "==" if g == e else "!="
            print(f"  Token {i}: pipeline={g} ({tokenizer.decode([g])!r}) {status} ref={e} ({tokenizer.decode([e])!r})")


if __name__ == "__main__":
    test_pipeline()
