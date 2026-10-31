"""Capture proxy: an OpenAI-compatible chat endpoint that sits between a harness and vLLM.

A harness only gets a base_url of the form http://host:port/r/<rollout_id>/v1. The proxy renders
the harness's messages with the policy's chat template, sends the *token ids* to vLLM, and records
exactly what the model saw and sampled (prompt ids, completion ids, per-token logprobs) under that
rollout id. The trainer never re-tokenizes text the model produced.
"""
import ast
import json
import re
import time
import uuid
from collections import defaultdict

import aiohttp
from aiohttp import web

TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
LFM_TOOL_RE = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.S)
_JSON_NAMES = {"true": True, "false": False, "null": None}


def _normalize_messages(messages):
    """OpenAI wire format -> what HF chat templates expect (tool-call arguments as dicts)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc["function"])
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        fn["arguments"] = {}
                calls.append({"type": "function", "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def _literal(node):
    """ast.literal_eval that also accepts the JSON true/false/null the LFM template emits for dicts/lists."""
    if isinstance(node, ast.Name) and node.id in _JSON_NAMES:
        return _JSON_NAMES[node.id]
    if isinstance(node, ast.List):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def _parse_pythonic(blob):
    """LFM2-style `[name(a='x', b=1), other()]` -> [(name, args)]; raises on anything malformed."""
    tree = ast.parse(blob.strip(), mode="eval").body
    calls = tree.elts if isinstance(tree, ast.List) else [tree]
    out = []
    for c in calls:
        if not isinstance(c, ast.Call) or not isinstance(c.func, ast.Name) or c.args:
            raise ValueError
        out.append((c.func.id, {kw.arg: _literal(kw.value) for kw in c.keywords}))
    return out


def _wrap(name, args):
    return {"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def parse_completion(text):
    """Split model output into (content, tool_calls, reasoning). Handles Qwen/Hermes `<tool_call>{json}`
    and LFM2 `<|tool_call_start|>[f(a=1)]<|tool_call_end|>`. Calls inside the thinking are ignored."""
    reasoning = ""
    if "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "").strip()
    calls = []
    for blob in TOOL_RE.findall(text):
        try:
            obj = json.loads(blob)
            name, args = obj["name"], obj.get("arguments", {})
            if not isinstance(args, dict):
                raise ValueError
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue  # malformed call: the harness sees no tool call, as a real server would
        calls.append(_wrap(name, args))
    for blob in LFM_TOOL_RE.findall(text):
        try:
            calls += [_wrap(n, a) for n, a in _parse_pythonic(blob)]
        except (SyntaxError, ValueError, TypeError):
            continue
    content = re.split(r"<tool_call>|<\|tool_call_start\|>", text)[0].strip() if calls else text.strip()
    return content, calls, reasoning


def _same_reply(sent, returned):
    """Did the harness echo back our reply unchanged (no repair, no rewrite)?"""
    if sent.get("role") != "assistant" or (sent.get("content") or "").strip() != returned["content"]:
        return False
    key = lambda calls: [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                         if isinstance(c["function"]["arguments"], str) else
                         (c["function"]["name"], c["function"]["arguments"]) for c in calls or []]
    try:
        return key(sent.get("tool_calls")) == key(returned.get("tool_calls"))
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


class CaptureProxy:
    def __init__(self, tokenizer, vllm_url="http://127.0.0.1:8000", port=8100, model_name=None,
                 max_tokens=1024, max_ctx=8192, temperature=1.0, template_kwargs=None):
        self.tok = tokenizer
        self.vllm_url = vllm_url
        self.port = port
        self.model_name = model_name      # which vLLM model/adapter the policy currently is
        self.max_tokens, self.max_ctx, self.temperature = max_tokens, max_ctx, temperature
        self.template_kwargs = template_kwargs or {}   # e.g. enable_thinking=False for Qwen3
        self.traces = defaultdict(list)   # rollout id -> list of captured calls
        self.last = {}                    # rollout id -> last request/reply, for token-level extension
        self.n_calls = self.n_extended = 0
        self.eos = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def base_url(self, rid):
        return f"http://127.0.0.1:{self.port}/r/{rid}/v1"

    def pop(self, rid):
        self.last.pop(rid, None)
        return self.traces.pop(rid, [])

    async def start(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600))
        app = web.Application(client_max_size=64 * 2**20)
        app.router.add_post("/r/{rid}/v1/chat/completions", self.chat)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", self.port).start()

    async def stop(self):
        await self.runner.cleanup()
        await self.session.close()

    def _render(self, messages, tools, gen):
        return self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=gen, tokenize=False,
                                            **self.template_kwargs)

    def _extend(self, rid, raw, messages, tools):
        """Token-in/token-out: if this request only appends messages after the reply we returned last
        time, the prompt is the previous prompt + the exact sampled ids + the template's suffix ids.
        Re-rendering would re-tokenize the model's own text (and Qwen3's template rewrites old
        assistant turns), so the trainer would learn from tokens the model never produced."""
        prev = self.last.get(rid)
        if prev is None or tools != prev["tools"]:
            return None
        k = len(prev["raw"])
        if len(raw) <= k + 1 or raw[:k] != prev["raw"] or not _same_reply(raw[k], prev["reply"]):
            return None
        full = self._render(messages, tools, True)
        n = self._render(messages[:k + 1], tools, False).count("<|im_start|>")
        starts = [i for i in range(len(full)) if full.startswith("<|im_start|>", i)]
        if len(starts) <= n or not full[:starts[n]].endswith("<|im_end|>\n"):
            return None
        return prev["ids"] + self.tok.encode(full[starts[n] - 1:], add_special_tokens=False)

    async def chat(self, request):
        rid = request.match_info["rid"]
        body = await request.json()
        messages = _normalize_messages(body["messages"])
        tools = body.get("tools") or None
        prompt_ids = self._extend(rid, body["messages"], messages, tools)
        self.n_extended += prompt_ids is not None
        self.n_calls += 1
        if prompt_ids is None:
            prompt_ids = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                                     return_dict=False, **self.template_kwargs)
        budget = min(self.max_tokens, self.max_ctx - len(prompt_ids))
        if budget < 32:
            return web.json_response({"error": {"message": "context length exceeded",
                                                "type": "invalid_request_error"}}, status=400)
        req = {"model": self.model_name, "prompt": prompt_ids, "max_tokens": budget,
               "temperature": self.temperature, "logprobs": 0, "skip_special_tokens": False}
        for attempt in range(3):   # a pooled keep-alive connection can be dropped under load; just resend
            try:
                async with self.session.post(f"{self.vllm_url}/v1/completions", json=req) as r:
                    if r.status != 200:
                        return web.json_response({"error": {"message": await r.text()}}, status=r.status)
                    out = (await r.json())["choices"][0]
                break
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError):
                if attempt == 2:
                    raise
        lp = out["logprobs"]
        comp_ids = [int(t.split(":", 1)[1]) for t in lp["tokens"]]
        self.traces[rid].append({"prompt_ids": prompt_ids, "completion_ids": comp_ids,
                                 "logprobs": lp["token_logprobs"], "finish_reason": out["finish_reason"]})
        text = out["text"].replace("<|im_end|>", "").replace("<|endoftext|>", "")
        content, calls, reasoning = parse_completion(text)
        if not tools and calls:   # like a real server: no tools in the request -> no tool parsing
            content, calls = text.split("</think>", 1)[-1].replace("<|tool_call_start|>", "").replace(
                "<|tool_call_end|>", "").strip(), []
        msg = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        if comp_ids and comp_ids[-1] == self.eos:   # only a cleanly finished turn can be extended
            self.last[rid] = {"raw": body["messages"], "tools": tools, "reply": msg,
                              "ids": prompt_ids + comp_ids}
        else:
            self.last.pop(rid, None)
        finish = "tool_calls" if calls else ("length" if out["finish_reason"] == "length" else "stop")
        return web.json_response({
            "id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
            "model": self.model_name, "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(comp_ids),
                      "total_tokens": len(prompt_ids) + len(comp_ids)}})
