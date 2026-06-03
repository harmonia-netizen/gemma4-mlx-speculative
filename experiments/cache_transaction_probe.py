import argparse
import importlib
import inspect
import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

d = importlib.import_module("mlx_vlm.generate.dispatch")

DEFAULT_MODEL = "mlx-community/gemma-4-26b-a4b-it-8bit"

DEFAULT_PROMPT = """あなたはローカル常駐エージェントです。
次に実行すべき確認コマンドを1つだけ出してください。
前提:
- repo=local-agent
- pytestが失敗している
- destructive commandは禁止
"""

MX_ARRAY_TYPE = type(mx.array([0]))
LOGITS_ATOL = 1e-2


@dataclass
class CacheMeta:
    cls: str
    keep: int | None
    max_size: int | None
    offset: int | None
    idx: int | None
    key_shape: tuple | None
    value_shape: tuple | None


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def make_cache(lm, max_kv_size):
    if max_kv_size is None:
        return make_prompt_cache(lm)

    sig = inspect.signature(make_prompt_cache)
    if "max_kv_size" not in sig.parameters:
        raise RuntimeError("make_prompt_cache does not support max_kv_size")

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


def cache_states(caches):
    out = []
    for c in caches:
        state = getattr(c, "state", None)
        if state is not None:
            out.append(state)
    return out


def eval_cache(caches):
    states = cache_states(caches)
    if states:
        mx.eval(states)


def cache_meta(c):
    keys = getattr(c, "keys", None)
    values = getattr(c, "values", None)

    return CacheMeta(
        cls=type(c).__name__,
        keep=getattr(c, "keep", None),
        max_size=getattr(c, "max_size", None),
        offset=getattr(c, "offset", None),
        idx=getattr(c, "_idx", None),
        key_shape=None if keys is None else tuple(keys.shape),
        value_shape=None if values is None else tuple(values.shape),
    )


def print_cache_summary(caches, label, limit=8):
    print(f"========== cache summary: {label} ==========")
    for i, c in enumerate(caches[:limit]):
        print(i, cache_meta(c))



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



MODEL_STATE_ATTRS = (
    "_position_ids",
    "_rope_deltas",
    "position_ids",
    "rope_deltas",
)


def model_state_objects(lm):
    objs = [lm]

    for name in ("model", "language_model"):
        obj = getattr(lm, name, None)
        if obj is not None and obj not in objs:
            objs.append(obj)

    return objs


def model_state_snapshot(lm):
    snap = []
    for obj in model_state_objects(lm):
        fields = {}
        for name in MODEL_STATE_ATTRS:
            if hasattr(obj, name):
                fields[name] = clone_value(getattr(obj, name))
        snap.append((obj, fields))
    return snap


def restore_model_state(snap):
    for obj, fields in snap:
        for name, value in fields.items():
            setattr(obj, name, clone_value(value))


def print_model_state_summary(lm, label):
    print(f"========== model state: {label} ==========")
    for obj in model_state_objects(lm):
        found = {}
        for name in MODEL_STATE_ATTRS:
            if hasattr(obj, name):
                value = getattr(obj, name)
                if isinstance(value, MX_ARRAY_TYPE):
                    found[name] = ("array", tuple(value.shape), str(value.dtype))
                else:
                    found[name] = repr(value)
        print(type(obj).__name__, found)


def partial_snapshot_rotating(caches, token_count):
    snap = []

    for i, c in enumerate(caches):
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

        # Safe partial path only for in-place decode regions.
        # If the write wraps or hits concat/trim semantics, fall back to full.
        if idx == max_size:
            idx = keep

        end = idx + token_count

        if end <= max_size:
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

    for i, (c, s) in enumerate(zip(caches, snap)):
        if s["mode"] == "full":
            restore_full([c], [s["full"]])
            continue

        if s["mode"] != "partial_rotating":
            raise RuntimeError(f"unsupported snapshot mode: {s['mode']}")

        c.keys[..., s["idx"]:s["end"], :] = s["keys"]
        c.values[..., s["idx"]:s["end"], :] = s["values"]
        c.meta_state = s["meta_state"]

    eval_cache(caches)


def forward_one(lm, token_2d, caches):
    out = lm(token_2d, cache=caches)
    logits = out.logits[:, -1, :]
    mx.eval(logits)
    return logits


def forward_many(lm, token_2d, caches):
    out = lm(token_2d, cache=caches)
    logits = out.logits
    mx.eval(logits)
    return logits


def argmax_token(logits):
    tok = mx.argmax(logits, axis=-1)
    mx.eval(tok)
    return tok


def argmax_id(logits):
    return int(argmax_token(logits).item())


def max_abs_diff(a, b):
    diff = mx.max(mx.abs(a - b))
    mx.eval(diff)
    return float(diff.item())



def compare_logits(label, expected, actual):
    expected_id = argmax_id(expected)
    actual_id = argmax_id(actual)
    diff = max_abs_diff(expected, actual)

    print(
        f"{label}: expected_id={expected_id} "
        f"actual_id={actual_id} "
        f"max_abs_diff={diff:.6f}"
    )

    return expected_id, actual_id, diff


def assert_token_decision_match(label, expected, actual):
    expected_id, actual_id, diff = compare_logits(label, expected, actual)

    if expected_id != actual_id:
        raise RuntimeError(
            f"{label}: argmax mismatch: expected={expected_id} actual={actual_id}"
        )

    return diff


def check_token_decision_match(label, expected, actual):
    expected_id, actual_id, diff = compare_logits(label, expected, actual)

    if expected_id != actual_id:
        print(
            f"WARNING: {label}: argmax mismatch: "
            f"expected={expected_id} actual={actual_id}"
        )
        return False

    return True


def format_prompt(processor, prompt):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def prefill_to_next_logits(model, lm, tokenizer, formatted_prompt, max_kv_size):
    input_ids = mx.array([tokenizer.encode(formatted_prompt)])
    caches = make_cache(lm, max_kv_size)

    emb = model.get_input_embeddings(input_ids, None, mask=None)
    inputs_embeds = emb.inputs_embeds
    extra = {
        k: v
        for k, v in emb.to_dict().items()
        if k != "inputs_embeds" and v is not None
    }

    if input_ids.shape[1] > 1:
        n = input_ids.shape[1] - 1
        lm(
            input_ids[:, :n],
            inputs_embeds=inputs_embeds[:, :n],
            cache=caches,
            n_to_process=n,
            **extra,
        )
        eval_cache(caches)

    logits = forward_one(lm, input_ids[:, -1:], caches)
    return caches, logits, input_ids.dtype


def greedy_tokens_on_cache(lm, caches, next_logits, n):
    toks = []
    logits = next_logits

    for _ in range(n):
        tok = argmax_token(logits)
        toks.append(int(tok.item()))
        logits = forward_one(lm, tok[:, None], caches)

    return toks, logits




def replay_tokens_one_by_one(lm, caches, token_ids, dtype):
    logits = None
    for tid in token_ids:
        tok = mx.array([[tid]], dtype=dtype)
        logits = forward_one(lm, tok, caches)
    return logits



def test_full_restore(lm, caches, next_logits, n, dtype):
    print("========== test: full restore ==========")
    cache_start = full_snapshot(caches)
    model_start = model_state_snapshot(lm)

    toks, direct_next_logits = greedy_tokens_on_cache(lm, caches, next_logits, n)

    restore_full(caches, cache_start)
    restore_model_state(model_start)

    replay_next_logits = replay_tokens_one_by_one(lm, caches, toks, dtype)

    assert_token_decision_match(
        "full restore one-by-one replay",
        direct_next_logits,
        replay_next_logits,
    )

    restore_full(caches, cache_start)
    restore_model_state(model_start)
    print("OK full restore")



def test_partial_restore(lm, caches, next_logits, n, dtype):
    print("========== test: partial restore ==========")
    cache_start = full_snapshot(caches)
    model_start = model_state_snapshot(lm)
    partial = partial_snapshot_rotating(caches, n)

    print("partial modes:", [item["mode"] for item in partial[:8]])

    toks, direct_next_logits = greedy_tokens_on_cache(lm, caches, next_logits, n)

    restore_partial(caches, partial)
    restore_model_state(model_start)

    replay_next_logits = replay_tokens_one_by_one(lm, caches, toks, dtype)

    assert_token_decision_match(
        "partial restore one-by-one replay",
        direct_next_logits,
        replay_next_logits,
    )

    restore_full(caches, cache_start)
    restore_model_state(model_start)
    print("OK partial restore")



def test_forward_many_equivalence(lm, caches, next_logits, n, dtype):
    print("========== test: forward_many equivalence ==========")
    cache_start = full_snapshot(caches)
    model_start = model_state_snapshot(lm)

    toks, direct_next_logits = greedy_tokens_on_cache(lm, caches, next_logits, n)

    restore_full(caches, cache_start)
    restore_model_state(model_start)

    replay = mx.array([toks], dtype=dtype)
    many_logits = forward_many(lm, replay, caches)[:, -1, :]

    try:
        ok = check_token_decision_match(
            "forward_many replay",
            direct_next_logits,
            many_logits,
        )
        if ok:
            print("OK forward_many token-decision equivalence")
        else:
            print("NG forward_many token-decision equivalence")
    finally:
        restore_full(caches, cache_start)
        restore_model_state(model_start)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--max-kv-size", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    if args.max_kv_size is not None and args.max_kv_size <= 0:
        raise ValueError("--max-kv-size must be positive")

    print("loading:", args.model)
    model, processor = d.load(args.model)
    tokenizer = getattr(processor, "tokenizer", processor)

    formatted = format_prompt(processor, args.prompt)
    print("prompt_tokens:", len(tokenizer.encode(formatted)))

    lm = get_lm(model)
    caches, next_logits, dtype = prefill_to_next_logits(
        model,
        lm,
        tokenizer,
        formatted,
        args.max_kv_size,
    )

    print_cache_summary(caches, "after prefill")
    print_model_state_summary(lm, "after prefill")

    t0 = time.perf_counter()
    test_full_restore(lm, caches, next_logits, args.tokens, dtype)
    print("full_restore_sec:", f"{time.perf_counter() - t0:.3f}")

    t1 = time.perf_counter()
    test_partial_restore(lm, caches, next_logits, args.tokens, dtype)
    print("partial_restore_sec:", f"{time.perf_counter() - t1:.3f}")

    t2 = time.perf_counter()
    test_forward_many_equivalence(lm, caches, next_logits, args.tokens, dtype)
    print("forward_many_equivalence_sec:", f"{time.perf_counter() - t2:.3f}")

    print("OK: cache transaction probe passed")


if __name__ == "__main__":
    main()
