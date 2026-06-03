import argparse
import importlib
import inspect
import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

d = importlib.import_module("mlx_vlm.generate.dispatch")

DEFAULT_TARGET_MODEL_PATH = "mlx-community/gemma-4-26b-a4b-it-8bit"
DEFAULT_DRAFT_MODEL_PATH = "mlx-community/gemma-4-e4b-it-4bit"

DEFAULT_PROMPT = """あなたはローカル常駐エージェントです。
次に実行すべき確認コマンドを1つだけ出してください。
前提:
- repo=local-agent
- pytestが失敗している
- destructive commandは禁止
"""

MX_ARRAY_TYPE = type(mx.array([0]))


@dataclass
class DecodeResult:
    text: str
    token_ids: list[int]
    elapsed_sec: float
    prefill_sec: float
    decode_sec: float
    tok_s: float
    accepted: int = 0
    drafted: int = 0
    rejected: int = 0


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def make_cache(lm, max_kv_size):
    if max_kv_size is None:
        return make_prompt_cache(lm)

    sig = inspect.signature(make_prompt_cache)
    if "max_kv_size" not in sig.parameters:
        raise RuntimeError("installed make_prompt_cache does not support max_kv_size")

    return make_prompt_cache(lm, max_kv_size=max_kv_size)


def clone_value(v):
    if isinstance(v, MX_ARRAY_TYPE):
        x = mx.array(v)
        mx.eval(x)
        return x
    if isinstance(v, tuple):
        return tuple(clone_value(x) for x in v)
    if isinstance(v, list):
        return [clone_value(x) for x in v]
    if isinstance(v, dict):
        return {k: clone_value(x) for k, x in v.items()}
    return v


def iter_slot_names(cls):
    names = []
    for base in reversed(cls.__mro__):
        slots = getattr(base, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for s in slots:
            if s not in ("__dict__", "__weakref__"):
                names.append(s)
    return names


def clone_cache_object(c, idx):
    nc = c.__class__.__new__(c.__class__)
    copied = False

    if hasattr(c, "__dict__"):
        for k, v in c.__dict__.items():
            setattr(nc, k, clone_value(v))
            copied = True

    for slot in iter_slot_names(type(c)):
        if hasattr(c, slot):
            setattr(nc, slot, clone_value(getattr(c, slot)))
            copied = True

    if not copied:
        raise TypeError(f"cache[{idx}] has no cloneable __dict__/__slots__: {type(c)}")

    return nc


def clone_cache_objects(prompt_cache):
    return [clone_cache_object(c, i) for i, c in enumerate(prompt_cache)]


def cache_states(prompt_cache):
    states = []
    for c in prompt_cache:
        state = getattr(c, "state", None)
        if state is not None:
            states.append(state)
    return states


def eval_cache(prompt_cache):
    states = cache_states(prompt_cache)
    if states:
        mx.eval(states)


def forward_one(lm, token_2d, prompt_cache):
    out = lm(token_2d, cache=prompt_cache)
    logits = out.logits[:, -1, :]
    mx.eval(logits)
    return logits


def forward_many(lm, token_2d, prompt_cache):
    out = lm(token_2d, cache=prompt_cache)
    logits = out.logits
    mx.eval(logits)
    return logits


def argmax_token(logits):
    token = mx.argmax(logits, axis=-1)
    mx.eval(token)
    return token


def argmax_id(logits):
    return int(argmax_token(logits).item())


def build_stop_ids(tokenizer):
    stop_texts = ["<eos>", "<turn|>", "<|turn>", "<channel|>", "<|channel>"]
    ids = set()

    for text in stop_texts:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(text)

        if len(encoded) == 1:
            ids.add(encoded[0])

    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.add(eos)

    return ids


def decode_text(tokenizer, token_ids):
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


def format_prompt(processor, prompt):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def encode_no_special(tokenizer, text):
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def assert_tokenizer_compatible(target_tokenizer, draft_tokenizer):
    probes = [
        "test",
        "pytest --tb=short",
        "次に実行すべき確認コマンドを1つだけ出してください。",
        "<turn|>",
        "<eos>",
    ]

    for probe in probes:
        target_ids = encode_no_special(target_tokenizer, probe)
        draft_ids = encode_no_special(draft_tokenizer, probe)

        if target_ids != draft_ids:
            raise RuntimeError(
                "target/draft tokenizer mismatch\n"
                f"probe={probe!r}\n"
                f"target={target_ids}\n"
                f"draft={draft_ids}"
            )


def assert_prompt_compatible(target_tokenizer, draft_tokenizer, target_prompt, draft_prompt):
    target_ids = target_tokenizer.encode(target_prompt)
    draft_ids_from_target_prompt = draft_tokenizer.encode(target_prompt)
    draft_ids_from_draft_prompt = draft_tokenizer.encode(draft_prompt)

    if target_ids != draft_ids_from_target_prompt:
        raise RuntimeError(
            "target formatted prompt token ids differ between tokenizers\n"
            f"target_len={len(target_ids)} draft_len={len(draft_ids_from_target_prompt)}"
        )

    if target_ids != draft_ids_from_draft_prompt:
        print("warning: target/draft chat_template differ; using target formatted prompt for both")


def prefill_to_last_prompt_token(wrapper_model, lm, tokenizer, formatted_prompt, max_kv_size):
    input_ids = mx.array([tokenizer.encode(formatted_prompt)])
    prompt_cache = make_cache(lm, max_kv_size)

    emb = wrapper_model.get_input_embeddings(input_ids, None, mask=None)
    inputs_embeds = emb.inputs_embeds
    extra = {
        k: v
        for k, v in emb.to_dict().items()
        if k != "inputs_embeds" and v is not None
    }

    start = time.perf_counter()

    if input_ids.shape[1] > 1:
        n = input_ids.shape[1] - 1
        lm(
            input_ids[:, :n],
            inputs_embeds=inputs_embeds[:, :n],
            cache=prompt_cache,
            n_to_process=n,
            **extra,
        )
        eval_cache(prompt_cache)

    return input_ids[:, -1:], prompt_cache, time.perf_counter() - start


def bootstrap_after_first_token(wrapper_model, lm, tokenizer, prompt, max_kv_size):
    cur, cache, prefill_sec = prefill_to_last_prompt_token(
        wrapper_model,
        lm,
        tokenizer,
        prompt,
        max_kv_size,
    )

    first_logits = forward_one(lm, cur, cache)
    first = argmax_token(first_logits)

    next_logits = forward_one(lm, first[:, None], cache)

    return first, next_logits, cache, prefill_sec


def bootstrap_draft_to_target_prefix(draft_model, draft_lm, tokenizer, prompt, first_token, max_kv_size):
    cur, cache, prefill_sec = prefill_to_last_prompt_token(
        draft_model,
        draft_lm,
        tokenizer,
        prompt,
        max_kv_size,
    )

    _ = forward_one(draft_lm, cur, cache)
    next_logits = forward_one(draft_lm, first_token[:, None], cache)

    return next_logits, cache, prefill_sec




def full_snapshot(caches):
    snapshot = []
    for c in caches:
        snapshot.append(
            {
                "keys": clone_value(getattr(c, "keys", None)),
                "values": clone_value(getattr(c, "values", None)),
                "meta_state": getattr(c, "meta_state", None),
                "offset": getattr(c, "offset", None),
                "_idx": getattr(c, "_idx", None),
            }
        )
    return snapshot


def restore_full(caches, snap):
    if len(caches) != len(snap):
        raise RuntimeError("cache/snapshot length mismatch")

    for i, (c, item) in enumerate(zip(caches, snap)):
        if hasattr(c, "keys"):
            c.keys = clone_value(item["keys"])
        if hasattr(c, "values"):
            c.values = clone_value(item["values"])

        if item["meta_state"] is not None:
            if not hasattr(c, "meta_state"):
                raise RuntimeError(f"cache[{i}] has no meta_state")
            c.meta_state = item["meta_state"]
        else:
            if item["offset"] is not None and hasattr(c, "offset"):
                c.offset = item["offset"]
            if item["_idx"] is not None and hasattr(c, "_idx"):
                c._idx = item["_idx"]

    eval_cache(caches)


def partial_snapshot_rotating(caches, token_count):
    snap = []

    for c in caches:
        keys = getattr(c, "keys", None)
        values = getattr(c, "values", None)
        meta_state = getattr(c, "meta_state", None)

        if keys is None or values is None or meta_state is None:
            snap.append({"mode": "full", "full": full_snapshot([c])[0]})
            continue

        cls_name = type(c).__name__
        if cls_name != "RotatingKVCache":
            snap.append({"mode": "full", "full": full_snapshot([c])[0]})
            continue

        idx = int(getattr(c, "_idx"))
        max_size = int(getattr(c, "max_size"))
        keep = int(getattr(c, "keep"))

        if token_count <= 0:
            raise RuntimeError("token_count must be positive")

        if idx == max_size:
            idx = keep

        end = idx + token_count
        current_size = int(keys.shape[2])

        # Safe fast path only when the write region is already allocated and
        # contiguous. If idx:end exceeds current_size, _update_in_place may grow
        # the backing arrays via concatenate; partial slice restore is then
        # unsafe and must fall back to full snapshot.
        if idx < current_size and end <= current_size and end <= max_size:
            key_slice = mx.array(keys[..., idx:end, :])
            value_slice = mx.array(values[..., idx:end, :])
            mx.eval(key_slice, value_slice)

            snap.append(
                {
                    "mode": "partial_rotating",
                    "meta_state": meta_state,
                    "idx": idx,
                    "end": end,
                    "keys": key_slice,
                    "values": value_slice,
                }
            )
        else:
            snap.append({"mode": "full", "full": full_snapshot([c])[0]})

    return snap


def restore_partial(caches, snap):
    if len(caches) != len(snap):
        raise RuntimeError("cache/snapshot length mismatch")

    for c, item in zip(caches, snap):
        if item["mode"] == "full":
            restore_full([c], [item["full"]])
            continue

        if item["mode"] != "partial_rotating":
            raise RuntimeError(f"unsupported snapshot mode: {item['mode']}")

        c.keys[..., item["idx"]:item["end"], :] = item["keys"]
        c.values[..., item["idx"]:item["end"], :] = item["values"]
        c.meta_state = item["meta_state"]

    eval_cache(caches)


def cache_transaction_snapshot(caches, token_count):
    return partial_snapshot_rotating(caches, token_count)


def cache_transaction_restore(caches, snapshot):
    restore_partial(caches, snapshot)



def run_target_greedy(target_model, tokenizer, prompt, stop_ids, max_tokens, max_kv_size):
    lm = get_lm(target_model)
    total_start = time.perf_counter()

    first, next_logits, cache, prefill_sec = bootstrap_after_first_token(
        target_model,
        lm,
        tokenizer,
        prompt,
        max_kv_size,
    )

    decode_start = time.perf_counter()
    out = []

    first_id = int(first.item())
    if first_id not in stop_ids:
        out.append(first_id)

    while len(out) < max_tokens:
        tok = argmax_token(next_logits)
        tid = int(tok.item())

        if tid in stop_ids:
            break

        out.append(tid)
        next_logits = forward_one(lm, tok[:, None], cache)

    decode_sec = time.perf_counter() - decode_start

    return DecodeResult(
        decode_text(tokenizer, out),
        out,
        time.perf_counter() - total_start,
        prefill_sec,
        decode_sec,
        len(out) / decode_sec if decode_sec > 0 else float("inf"),
    )


def commit_tokens_to_cache(lm, prompt_cache, token_ids, dtype):
    if not token_ids:
        return None

    arr = mx.array([token_ids], dtype=dtype)
    logits = forward_many(lm, arr, prompt_cache)
    eval_cache(prompt_cache)
    return logits[:, -1, :]


def run_speculative(
    target_model,
    draft_model,
    tokenizer,
    prompt,
    stop_ids,
    max_tokens,
    block_size,
    max_kv_size,
):
    target_lm = get_lm(target_model)
    draft_lm = get_lm(draft_model)

    total_start = time.perf_counter()

    first, target_next_logits, target_cache, target_prefill = bootstrap_after_first_token(
        target_model,
        target_lm,
        tokenizer,
        prompt,
        max_kv_size,
    )

    draft_next_logits, draft_cache, draft_prefill = bootstrap_draft_to_target_prefix(
        draft_model,
        draft_lm,
        tokenizer,
        prompt,
        first,
        max_kv_size,
    )

    decode_start = time.perf_counter()

    out = []
    first_id = int(first.item())

    if first_id in stop_ids:
        decode_sec = time.perf_counter() - decode_start
        return DecodeResult(
            "",
            [],
            time.perf_counter() - total_start,
            target_prefill + draft_prefill,
            decode_sec,
            0.0,
        )

    out.append(first_id)

    accepted = 0
    drafted = 0
    rejected = 0

    while len(out) < max_tokens:
        remaining = max_tokens - len(out)
        if remaining <= 0:
            break

        proposal_limit = min(block_size, remaining)

        proposal_ids = []
        cur_draft_logits = draft_next_logits

        target_snapshot = full_snapshot(target_cache)
        draft_snapshot = cache_transaction_snapshot(draft_cache, proposal_limit)

        for _ in range(proposal_limit):
            dtok = argmax_token(cur_draft_logits)
            did = int(dtok.item())
            proposal_ids.append(did)
            drafted += 1

            cur_draft_logits = forward_one(draft_lm, dtok[:, None], draft_cache)

            if did in stop_ids:
                break

        if not proposal_ids:
            break

        proposal = mx.array([proposal_ids], dtype=first.dtype)

        verify_logits = forward_many(target_lm, proposal, target_cache)

        emitted = []
        mismatch = False
        reached_stop = False

        for i, did in enumerate(proposal_ids):
            logits_i = target_next_logits if i == 0 else verify_logits[:, i - 1, :]
            target_tok = argmax_token(logits_i)
            target_id = int(target_tok.item())

            if target_id == did:
                chosen_id = did
                accepted += 1
            else:
                chosen_id = target_id
                mismatch = True
                rejected += 1

            if chosen_id in stop_ids:
                reached_stop = True
                break

            emitted.append(chosen_id)

            if len(out) + len(emitted) >= max_tokens:
                break

            if mismatch:
                break

        restore_full(target_cache, target_snapshot)
        cache_transaction_restore(draft_cache, draft_snapshot)

        if not emitted:
            break

        out.extend(emitted)

        target_next_logits = commit_tokens_to_cache(
            target_lm,
            target_cache,
            emitted,
            first.dtype,
        )
        draft_next_logits = commit_tokens_to_cache(
            draft_lm,
            draft_cache,
            emitted,
            first.dtype,
        )

        if target_next_logits is None or draft_next_logits is None:
            break

        mx.eval(target_next_logits, draft_next_logits)

        if reached_stop:
            break

    decode_sec = time.perf_counter() - decode_start

    return DecodeResult(
        decode_text(tokenizer, out),
        out,
        time.perf_counter() - total_start,
        target_prefill + draft_prefill,
        decode_sec,
        len(out) / decode_sec if decode_sec > 0 else float("inf"),
        accepted,
        drafted,
        rejected,
    )


def print_result(name, r):
    print(f"========== {name} ==========")
    print(r.text)
    print("tokens:", len(r.token_ids))
    print("elapsed_sec:", f"{r.elapsed_sec:.3f}")
    print("prefill_sec:", f"{r.prefill_sec:.3f}")
    print("decode_sec:", f"{r.decode_sec:.3f}")
    print("decode_tok_s:", f"{r.tok_s:.3f}")

    if r.drafted:
        print("accepted:", r.accepted, "/", r.drafted, f"({r.accepted / r.drafted:.1%})")
        print("rejected:", r.rejected)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL_PATH)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL_PATH)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--draft-block-size", type=int, default=2)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.draft_block_size <= 0:
        raise ValueError("--draft-block-size must be positive")
    if args.max_kv_size is not None and args.max_kv_size <= 0:
        raise ValueError("--max-kv-size must be positive")

    print("loading target:", args.target_model)
    target_model, processor = d.load(args.target_model)

    print("loading draft:", args.draft_model)
    draft_model, draft_processor = d.load(args.draft_model)

    tokenizer = getattr(processor, "tokenizer", processor)
    draft_tokenizer = getattr(draft_processor, "tokenizer", draft_processor)

    assert_tokenizer_compatible(tokenizer, draft_tokenizer)

    target_prompt = format_prompt(processor, args.prompt)
    draft_prompt = format_prompt(draft_processor, args.prompt)
    assert_prompt_compatible(tokenizer, draft_tokenizer, target_prompt, draft_prompt)

    prompt_tokens = len(tokenizer.encode(target_prompt))
    print("prompt_tokens:", prompt_tokens)

    stop_ids = build_stop_ids(tokenizer)
    print("stop_ids:", sorted(stop_ids))

    greedy = run_target_greedy(
        target_model,
        tokenizer,
        target_prompt,
        stop_ids,
        args.max_tokens,
        args.max_kv_size,
    )

    spec = run_speculative(
        target_model,
        draft_model,
        tokenizer,
        target_prompt,
        stop_ids,
        args.max_tokens,
        args.draft_block_size,
        args.max_kv_size,
    )

    print_result("target greedy", greedy)
    print_result("speculative", spec)

    print("========== check ==========")
    if spec.token_ids != greedy.token_ids:
        print("MISMATCH")
        print("greedy ids:", greedy.token_ids)
        print("spec ids:  ", spec.token_ids)
        print("greedy text:", greedy.text)
        print("spec text:", spec.text)
        raise SystemExit(2)

    print("OK: speculative output matches target greedy")


if __name__ == "__main__":
    main()
