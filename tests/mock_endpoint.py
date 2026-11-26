# -*- coding: utf-8 -*-
"""本地假端点：用来验证 modelcheck 的探针逻辑本身对不对。

它不联网、不需要密钥，可以在测试里精确复现四种行为：

  real        诚实的端点：词表是 o200k_base、功能齐全、能力题全对
  downgrade   降级端点：声称 gpt-4o，实际走 cl100k_base（gpt-4 那一代）
  fake        套壳端点：词表对不上任何已知参照、功能缺失、能力题答错
  recomputed  usage 被端点自己重算：han_dup 不等于 50，指纹探针应当自检失败

这个文件只用于开发和测试，不是工具的一部分。
"""

import json
import re
import time
from http.server import BaseHTTPRequestHandler

# 与 modelcheck.FP_PROBES 保持一致。测试会校验这一点。
FP_PROBES = [
    ("han_dup", "测试" * 50),
    ("han_rare", "龘靐齉爩鱻麤" * 20),
    ("emoji", "🦙🐉🌏" * 20),
    ("real_cn", "多轮对话的成本优化，大部分文章都在讲同一件事：裁剪历史记录。" * 3),
    ("mix", "API 网关和 token 预算" * 10),
]

# 实测值（tiktoken 0.14.0），即「扣掉基线 ok 之后的增量」
REF = {
    "o200k_base": {"han_dup": 50, "han_rare": 260, "emoji": 140, "real_cn": 69, "mix": 70},
    "cl100k_base": {"han_dup": 50, "han_rare": 300, "emoji": 180, "real_cn": 99, "mix": 110},
    # 2026-11-26 通过一个真实网关实测得到（见 modelcheck.VOCAB_REF 的说明）
    "anthropic_claude": {"han_dup": 100, "han_rare": 260, "emoji": 180, "real_cn": 102, "mix": 101},
    "glm_zhipu": {"han_dup": 50, "han_rare": 240, "emoji": 100, "real_cn": 54, "mix": 110},
}

# 基线 "ok" 在各词表下都当作 1 个 token。
# 数值本身不重要（差分会把固定开销抵消），重要的是三套参照用同一个基数。
COUNT_OK = {"o200k_base": 1, "cl100k_base": 1, "anthropic_claude": 1, "glm_zhipu": 1}

# 「ok」+ 探针文本  →  {词表: 整段文本单独编码出的 token 数}
#
# ⚠ 注意语义：REF 里存的是「比基线多出多少」（即差分值），
#   而这张表要的是「整段文本有多少」。所以这里要加回基线的 1。
_COUNT_BY_TEXT = {v: {} for v in REF}
for _vocab in REF:
    _COUNT_BY_TEXT[_vocab]["ok"] = COUNT_OK[_vocab]
    for _key, _text in FP_PROBES:
        _COUNT_BY_TEXT[_vocab]["ok" + _text] = REF[_vocab][_key] + COUNT_OK[_vocab]


def _lookup(text, vocab):
    return _COUNT_BY_TEXT.get(vocab, {}).get(text)

# 能力题标准答案（与 modelcheck.CAPABILITY_QUESTIONS 一致）
ANSWERS = {
    "847": "33033",
    "反转": "f",
    "101101": "45",
    "向右走 3 步": "10",
    "f(f(1))": "17",
    "banana": "3",
    "x[1::2]": "6",
    "站成一排": "BAC",
    "第 5 个质数": "11",
    "长方体": "60",
}

MESSAGE_OVERHEAD = 4
PRIMING = 3


def _han(text):
    return sum(1 for c in text if "\u4e00" <= c <= "\u9fff")


def count_tokens(text, vocab):
    """模拟后端分词器。已知探针查表，未知文本用启发式（确定性，不随机）。"""
    hit = _lookup(text, vocab)
    if hit is not None:
        return hit
    h = _han(text)
    other = len(text) - h
    if vocab == "o200k_base":
        return int(round(h * 0.85)) + int(round(other * 0.30)) + 1
    if vocab == "cl100k_base":
        return int(round(h * 1.20)) + int(round(other * 0.32)) + 1
    if vocab == "anthropic_claude":
        return int(round(h * 1.00)) + int(round(other * 0.28)) + 1
    if vocab == "glm_zhipu":
        return int(round(h * 0.52)) + int(round(other * 0.26)) + 1
    # fake：假装是别家词表，比值既不像 o200k 也不像 cl100k
    return int(round(h * 1.05)) + int(round(other * 0.42)) + 1


def count_recomputed(text):
    """端点自己重算 usage 时的行为：拿字符数除一个系数，不看词表。"""
    return max(1, len(text) // 2)


def _answer(question, mode):
    """按题目关键词返回答案。"""
    for key, val in ANSWERS.items():
        if key in question:
            if mode == "fake":
                # 套壳端点答错：返回一个看起来像但不对的答案
                if val.isdigit():
                    return str(int(val) + 7)
                if val == "f":
                    return "d"
                if val == "BAC":
                    return "ACB"
            return val
    return "我不确定"


class Handler(BaseHTTPRequestHandler):
    mode = "real"

    # 关掉默认日志，测试输出干净
    def log_message(self, fmt, *args):
        pass

    # -- 工具 --------------------------------------------------------------
    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, message, etype="invalid_request_error"):
        self._send_json(code, {"error": {"message": message, "type": etype}})

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    # -- 路由 --------------------------------------------------------------
    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._err(404, "no such endpoint: " + self.path, "not_found_error")
            return

        auth = self.headers.get("Authorization") or ""
        if not auth.startswith("Bearer ") or len(auth) < 12 or "invalid" in auth:
            self._err(401, "Incorrect API key provided", "authentication_error")
            return

        body = self._read_body()
        model = body.get("model") or ""
        if model.startswith("no-such-model") or model not in self.allowed_models():
            self._err(404, "The model `%s` does not exist" % model, "not_found_error")
            return

        # strict_temp 模式：模拟只接受 temperature=1 或不传的模型
        # （实测 kimi-k3.1-pro 就是这样，传 0 / 0.2 / 0.5 全部 400）
        if self.mode == "strict_temp":
            t = body.get("temperature", None)
            if t is not None and float(t) != 1.0:
                self._err(400, "bad response status code 400", "invalid_request_error")
                return

        messages = body.get("messages") or []
        prompt_text = "\n".join(
            str(m.get("content") or "") for m in messages if isinstance(m, dict)
        )
        user_text = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                user_text = str(m.get("content") or "")

        if self.mode == "recomputed":
            pt = count_recomputed(prompt_text) + MESSAGE_OVERHEAD * len(messages) + PRIMING
        else:
            vocab = {
                "downgrade": "cl100k_base",
                "claude": "anthropic_claude",
                "glm": "glm_zhipu",
                "fake": "other",
                "strict_temp": "o200k_base",
            }.get(self.mode, "o200k_base")
            pt = (
                count_tokens(prompt_text, vocab)
                + MESSAGE_OVERHEAD * len(messages)
                + PRIMING
            )

        if body.get("stream"):
            self._stream(body, user_text, pt, model)
            return

        text = self._content(body, user_text)
        ct = max(1, count_tokens(text, "o200k_base"))
        resp = {
            "id": "chatcmpl-mock" + str(int(time.time() * 1000) % 100000),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
            },
        }

        if body.get("logprobs") and self.mode in ("real", "downgrade"):
            resp["choices"][0]["logprobs"] = {
                "content": [
                    {"token": text[:1] or "a", "logprob": -0.1, "bytes": None, "top_logprobs": []}
                ]
            }

        if body.get("tools") and self.mode in ("real", "downgrade"):
            resp["choices"][0]["message"] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_mock1",
                        "type": "function",
                        "function": {
                            "name": body["tools"][0]["function"]["name"],
                            "arguments": '{"city": "Beijing"}',
                        },
                    }
                ],
            }

        # max_tokens 尊重（只影响 completion 计数口径）
        self._send_json(200, resp)

    def allowed_models(self):
        return (
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-5",
            "o3",
            "claude-sonnet-4-5",
            "claude-fable-5-2",
            "claude-opus-4",
            "glm-5.3-flash",
        )

    def _content(self, body, user_text):
        # JSON 模式
        rf = body.get("response_format") or {}
        if isinstance(rf, dict) and rf.get("type") == "json_object":
            if self.mode == "fake":
                return "当然可以！这是一个 JSON：ok true"
            return '{"ok": true}'

        # stop 序列：真端点遇到 stop 就截断，套壳端点不理会它
        stops = body.get("stop")
        stop_list = stops if isinstance(stops, list) else ([stops] if stops else [])
        if "AAAA" in stop_list:
            if self.mode in ("real", "downgrade"):
                return "AAA"          # 在第一个 AAAA 之前就停下
            return "A" * 40           # 完全不理 stop，输出里一定有 AAAA

        # max_tokens：真端点会严格遵守，套壳端点经常忽略
        if "TCP" in user_text:
            if self.mode == "fake":
                return (
                    "TCP 三次握手是建立可靠连接的过程，包含 SYN、SYN-ACK、ACK 三个步骤，"
                    "客户端和服务器各自确认收发能力。" * 2
                )
            return "三次握手"

        # 能力题
        if any(k in user_text for k in ANSWERS):
            return _answer(user_text, self.mode)

        # 自我介绍（流式探针用）
        if "介绍你自己" in user_text:
            base = "我是一个用于测试的语言模型端点，回答尽量简洁。"
            if self.mode == "fake":
                return base * 6
            return base * 3

        # seed 可复现 / 数到 5
        if "数到 5" in user_text:
            return "1, 2, 3, 4, 5"

        return "pong"

    def _stream(self, body, user_text, pt, model):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        text = self._content(body, user_text)
        # 按块发，块之间留间隔，让「首块时间 < 总耗时」这一点可测
        step = max(1, len(text) // 6)
        chunks = [text[i : i + step] for i in range(0, len(text), step)] or [text]

        for i, ch in enumerate(chunks):
            if i > 0:
                time.sleep(0.05)
            piece = {
                "id": "chatcmpl-mockstream",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {"content": ch}, "finish_reason": None}],
            }
            self.wfile.write(
                ("data: " + json.dumps(piece, ensure_ascii=False) + "\n\n").encode("utf-8")
            )
            self.wfile.flush()

        tail = {
            "id": "chatcmpl-mockstream",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": 10, "total_tokens": pt + 10},
        }
        self.wfile.write(
            ("data: " + json.dumps(tail, ensure_ascii=False) + "\n\n").encode("utf-8")
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def make_handler(mode="real"):
    cls = type("Handler_" + mode, (Handler,), {"mode": mode})
    return cls
