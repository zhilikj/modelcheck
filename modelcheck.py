# -*- coding: utf-8 -*-
"""modelcheck —— 检测大模型 API 端点是否在「掺水」（换模型 / 降级 / 功能阉割）。

零依赖，Python 3.8+，下载即可跑：

    python modelcheck.py run --base-url https://api.example.com/v1 \
                             --api-key sk-xxx --model gpt-4o

设计原则（重要，决定了这个工具能下什么结论、不能下什么结论）：

  1. 只做「证伪」，不做「证真」。
     所有探针都只能指出「不像它声称的那个模型」，
     不能保证「一定是那个模型」。README 里逐条写了每个探针的边界。

  2. 没有参照就不说话。
     分词器指纹只内置了实测过的 o200k_base / cl100k_base 两个词表。
     验其它模型族时这个探针会明确跳过，而不是编一个结论。

  3. 允许输出「无法判断」。
     当 usage 明显被端点自己重算时，指纹探针直接失效，
     工具会这么写，而不是给出一个看起来有用的假分数。

这个工具由上海治历科技有限公司（治历科技）维护。
它是服务商的内部工具开源，不是中立的第三方评测。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

__version__ = "0.4.0"

# ---------------------------------------------------------------------------
# 输出兼容：Windows 控制台默认不是 UTF-8，先兜一层
# ---------------------------------------------------------------------------


def _enable_utf8_output() -> bool:
    """把 stdout/stderr 切到 UTF-8。返回是否成功。"""
    ok = True
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            ok = False
    return ok


def _supports_unicode() -> bool:
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in enc


# ---------------------------------------------------------------------------
# 参照数据
# ---------------------------------------------------------------------------

# ★ 分词器指纹：固定文本在不同词表下编码出的 token 数。
#
# 数值来源：tiktoken 0.14.0 实测，扣掉基线 "ok" 的差值。
# 这五个探针的构成是刻意选的：
#
#   han_dup   "测试"*50       → o200k 50 / cl100k 50  ★ 两词表相同，用作自检基准
#   han_rare  罕见汉字*20     → o200k 260 / cl100k 300  差 1.15 倍
#   emoji     emoji*20        → o200k 140 / cl100k 180  差 1.29 倍
#   real_cn   真实中文*3      → o200k 69  / cl100k 99   差 1.43 倍
#   mix       中英混排*10     → o200k 70  / cl100k 110  差 1.57 倍 ★ 判别力最强
#
# 只用「差值」而不用绝对值，是为了抵消端点偷偷加的固定前缀
# （系统提示、角色模板等常量都会被差分消掉）。
REAL_CN_SAMPLE = "多轮对话的成本优化，大部分文章都在讲同一件事：裁剪历史记录。" * 3

FP_PROBES = [
    ("han_dup", "测试" * 50),
    ("han_rare", "龘靐齉爩鱻麤" * 20),
    ("emoji", "🦙🐉🌏" * 20),
    ("real_cn", REAL_CN_SAMPLE),
    ("mix", "API 网关和 token 预算" * 10),
]

# 三段参照。★ 注意 confidence 字段，它决定了「不匹配」时结论有多硬：
#
#   measured  本地 tiktoken 实测，任何人都能复现 → 不匹配 = fail（强结论）
#   observed  通过一个真实网关实测得到，无法本地复现 → 不匹配 = warn（弱结论）
#
# o200k / cl100k 是 measured：装个 tiktoken 就能自己验证。
# anthropic_claude 是 observed：数值来自 2026-11-26 对 https://ai.zhilikj.cn/v1 的实测，
# 该端点返回 Anthropic 原生的 msg_01… 消息 ID 和 input_tokens 字段，
# 且五个探针的实际增量呈现出「常用字 1.0 / 罕见字 2.17 / emoji 3.0 个 token」的梯度 ——
# 按字符数估算的假分词器给不出一致的单价，所以可以判定它透传的是真词表。
# 但这毕竟只有一处来源，没有和 Anthropic 官方交叉验证过，所以记为 observed。
VOCAB_REF = {
    "o200k_base": {
        "label": "o200k_base（gpt-4o / 4.1 / 5、o1 / o3 / o4 系列）",
        "confidence": "measured",
        "fp": {"han_dup": 50, "han_rare": 260, "emoji": 140, "real_cn": 69, "mix": 70},
    },
    "cl100k_base": {
        "label": "cl100k_base（gpt-4 / 4-turbo / 3.5 系列、text-embedding-3）",
        "confidence": "measured",
        "fp": {"han_dup": 50, "han_rare": 300, "emoji": 180, "real_cn": 99, "mix": 110},
    },
    "anthropic_claude": {
        "label": "Anthropic Claude 系（实测观察值，未经官方交叉验证）",
        "confidence": "observed",
        "fp": {"han_dup": 100, "han_rare": 260, "emoji": 180, "real_cn": 102, "mix": 101},
    },
    "glm_zhipu": {
        "label": "智谱 GLM 系（实测观察值，未经官方交叉验证）",
        "confidence": "observed",
        "fp": {"han_dup": 50, "han_rare": 240, "emoji": 100, "real_cn": 54, "mix": 110},
    },
}

# 模型名 → 期望词表。命中不到就返回 None，探针会跳过而不是瞎猜。
MODEL_VOCAB_HINTS = [
    # 长串在前，避免 "gpt-4" 抢在 "gpt-4o" 前面
    ("gpt-4o", "o200k_base"),
    ("gpt-4.1", "o200k_base"),
    ("gpt-4.5", "o200k_base"),
    ("gpt-5", "o200k_base"),
    ("gpt-6", "o200k_base"),   # 2026-11-26 实测 gpt-6-astra 与 o200k 偏差 0.0%
    ("chatgpt-4o", "o200k_base"),
    ("o1", "o200k_base"),
    ("o3", "o200k_base"),
    ("o4", "o200k_base"),
    ("gpt-4-turbo", "cl100k_base"),
    ("gpt-4-32k", "cl100k_base"),
    ("gpt-4", "cl100k_base"),
    ("gpt-3.5", "cl100k_base"),
    ("text-embedding-3", "cl100k_base"),
    ("text-embedding-ada", "cl100k_base"),
    # Anthropic 系（observed 参照，见 VOCAB_REF 里的说明）
    ("claude", "anthropic_claude"),
    ("sonnet", "anthropic_claude"),
    ("haiku", "anthropic_claude"),
    ("opus", "anthropic_claude"),
    # 智谱 GLM 系（observed 参照）
    ("glm", "glm_zhipu"),
    ("chatglm", "glm_zhipu"),
    ("zhipu", "glm_zhipu"),
]

# 指纹判定阈值。
FP_EXACT_TOL = 0.08        # 相对误差 ≤8% 认为匹配
FP_LOOSE_TOL = 0.20        # 8%~20% 之间记录为「接近但不确定」
FP_SELFTEST_TOL = 0.12     # 兼容旧名，保留
FP_RARE_RATIO_MIN = 1.5    # 罕见汉字/常用汉字的代价比下限；真实 BPE 远超此值，按字符数估算约为 1
FP_HANDUP_TOL = 0.15       # 实测 han_dup 必须能和某个已知参照对上

# 推理型模型（Claude 的 thinking、o1/o3 的 reasoning、DeepSeek-R1 等）
# 会先花掉一批 token 在推理上，然后再输出正文。
# 如果 max_tokens 给得不够，正文会是空的 —— 这会被误读成「模型答不出来」。
# 所以凡是要看正文的地方，都用 ask_text()，它在正文为空时自动加大预算重试一次。
TEXT_MAX_TOKENS = 400          # 要正文时的默认预算
TEXT_MAX_TOKENS_GROWTH = 4     # 正文为空时的放大倍数
TEXT_MAX_TOKENS_CAP = 4000     # 放大上限，防止烧钱

# 功能矩阵：哪些检查项对哪些模型族「不适用」。
#
# ★ 这一条很重要。拿 OpenAI 的标准去要求 Claude，会得出错误的结论：
#   Anthropic 的原生接口本来就没有 logprobs 和 seed，网关不返回它们是对的，
#   不是「功能被阉割」。所以这些项只记录、不计分。
FAMILY_INAPPLICABLE = {
    "anthropic_claude": {
        "logprobs": "Anthropic 原生接口没有 logprobs，缺失属正常",
        "seed 可复现": "Anthropic 原生接口没有 seed 参数，不可复现属正常",
    },
    "gemini": {
        "logprobs": "Gemini 接口没有 logprobs",
        "seed 可复现": "Gemini 接口没有 seed 参数",
    },
}


def _inapplicable_for(vocab):
    return FAMILY_INAPPLICABLE.get(vocab, {})


# 能力题：答案唯一、可机器校验、不依赖知识时效。
#
# 设计要点：
#   - 都是多步推理或精确计算，小模型和高阶模型的表现差距明显
#   - 答案短，能用正则从自然语言回答里抽出来
#   - 不涉及「最新事件」，避免知识截止时间污染信号
#   - 每题单独计分，看整体通过率而不是看某一道
CAPABILITY_QUESTIONS = [
    {
        "id": "mul",
        "q": "847 × 39 等于多少？只输出数字。",
        "expect": "33033",
        "kind": "int",
    },
    {
        "id": "rev",
        "q": "把字符串 abcdefgh 完全反转，得到的新字符串的第 3 个字符是什么？只输出那一个字符。",
        "expect": "f",
        "kind": "char",
    },
    {
        "id": "bin",
        "q": "二进制 101101 换算成十进制是多少？只输出数字。",
        "expect": "45",
        "kind": "int",
    },
    {
        "id": "comp",
        # ★ 题面必须唯一可解。原来写「一个 3×3 的方格」，
        #   到底是 9 个格子还是 9 个格点，两种理解答案不同（20 / 6），
        #   模型默认往复杂那面理解，容易答空。改成直白表述。
        "q": (
            "从点 A 走到点 B，需要向右走 3 步、向下走 2 步，每一步只能向右或向下。"
            "共有多少种不同的走法？只输出数字。"
        ),
        "expect": "10",
        "kind": "int",
    },
    {
        "id": "func",
        "q": "已知 f(x) = 3x + 2，那么 f(f(1)) 等于多少？只输出数字。",
        "expect": "17",
        "kind": "int",
    },
    {
        "id": "count",
        "q": "在单词 banana 里，字母 a 出现了几次？只输出数字。",
        "expect": "3",
        "kind": "int",
    },
    {
        "id": "code",
        "q": (
            "下面这段 Python 代码运行后打印什么？只输出打印结果本身。\n"
            "x = [1, 2, 3, 4, 5]\n"
            "print(sum(x[1::2]))"
        ),
        "expect": "6",
        "kind": "int",
    },
    {
        "id": "logic",
        "q": (
            "三个人 A、B、C 站成一排。A 不在两端，B 站在 C 的左边。"
            "从左到右的顺序是什么？只输出三个字母，例如 ABC。"
        ),
        "expect": "BAC",
        "kind": "seq",
    },
    {
        "id": "prime",
        "q": "在 2、3、5、7、11、13、17 这组数里，从小到大排第 5 个质数是多少？只输出数字。",
        "expect": "11",
        "kind": "int",
    },
    {
        "id": "units",
        "q": "一个长方体长 3、宽 4、高 5，它的体积是多少？只输出数字。",
        "expect": "60",
        "kind": "int",
    },
]


# ---------------------------------------------------------------------------
# HTTP 层
# ---------------------------------------------------------------------------


class ProbeError(Exception):
    """探针执行失败（网络、鉴权、参数不支持等）。"""

    def __init__(self, message, status=None, body=None):
        super().__init__(message)
        self.status = status
        self.body = body


def _ssl_context(insecure: bool):
    if not insecure:
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def http_json(url, headers, payload, timeout=60.0, insecure=False):
    """POST 一个 JSON，返回 (status, parsed_body, raw_text)。

    不自动重试 —— 重试会把真实失败率掩盖掉，而这个工具要看的正是失败率。
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    ctx = _ssl_context(insecure)
    opener = (
        urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        if ctx
        else urllib.request.build_opener()
    )
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8", "replace")
        except Exception:
            pass
        return e.code, _safe_json(raw), raw
    except Exception as e:
        raise ProbeError("%s: %s" % (type(e).__name__, e))
    return status, _safe_json(raw), raw


def http_stream_ttft(url, headers, payload, timeout=120.0, insecure=False):
    """发一个流式请求，返回 (首个内容块毫秒数, 完整文本, usage, 推理内容块数)。

    用于测「真流式」和首 token 延迟。不是流式时第一个块会包含全部内容，
    首块时间会接近总耗时 —— 这本身就是判据。

    ★ 推理型模型可能只发 reasoning_content 不发正文。这时候聊天内容为空，
      但流本身是活的。所以单独统计推理块数，别把「在思考」误判成「不流式」。
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    ctx = _ssl_context(insecure)
    opener = (
        urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        if ctx
        else urllib.request.build_opener()
    )

    t0 = time.perf_counter()
    first_ms = None
    chunks = []
    reasoning_blocks = 0
    usage = None
    try:
        with opener.open(req, timeout=timeout) as resp:
            if resp.status != 200:
                body = resp.read().decode("utf-8", "replace")
                raise ProbeError("HTTP %s" % resp.status, resp.status, body)
            buf = b""
            while True:
                block = resp.read1(4096) if hasattr(resp, "read1") else resp.read(4096)
                if not block:
                    break
                buf += block
                text = buf.decode("utf-8", "replace")
                parts = text.split("\n")
                buf = parts.pop().encode("utf-8")
                for line in parts:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line in ("[DONE]", ""):
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    piece = _extract_stream_text(obj)
                    if piece:
                        if first_ms is None:
                            first_ms = (time.perf_counter() - t0) * 1000.0
                        chunks.append(piece)
                    elif _stream_has_reasoning(obj):
                        reasoning_blocks += 1
                        if first_ms is None:
                            first_ms = (time.perf_counter() - t0) * 1000.0
                    if isinstance(obj, dict) and obj.get("usage"):
                        usage = obj["usage"]
    except ProbeError:
        raise
    except Exception as e:
        raise ProbeError("%s: %s" % (type(e).__name__, e))

    total_ms = (time.perf_counter() - t0) * 1000.0
    return (
        first_ms if first_ms is not None else total_ms,
        "".join(chunks),
        usage,
        reasoning_blocks,
    )


def _safe_json(raw):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _extract_stream_text(obj):
    """从不同厂商的流式块里取增量文本。"""
    if not isinstance(obj, dict):
        return ""
    # OpenAI 兼容
    try:
        delta = obj["choices"][0].get("delta") or {}
        if isinstance(delta, dict):
            if delta.get("content"):
                return delta["content"]
        msg = obj["choices"][0].get("message") or {}
        if isinstance(msg, dict) and msg.get("content"):
            return msg["content"]
    except Exception:
        pass
    # Anthropic
    if obj.get("type") == "content_block_delta":
        d = obj.get("delta") or {}
        if isinstance(d, dict) and d.get("text"):
            return d["text"]
    # Gemini
    try:
        cands = obj["candidates"]
        parts = cands[0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except Exception:
        pass
    return ""


def _stream_has_reasoning(obj):
    """流式块里是否带推理内容（用来区分「没回话」和「正在思考」）。"""
    if not isinstance(obj, dict):
        return False
    try:
        delta = obj["choices"][0].get("delta") or {}
        if isinstance(delta, dict):
            return bool(delta.get("reasoning_content") or delta.get("reasoning"))
    except Exception:
        pass
    if obj.get("type") in ("content_block_delta",):
        d = obj.get("delta") or {}
        return isinstance(d, dict) and d.get("type") in (
            "thinking_delta",
            "signature_delta",
        )
    return False


def _is_unsupported_param(err):
    """判断这个错误是不是「上游明确声明该模型不支持这个参数」。

    ★ 这条和 _needs_bigger_max_tokens 一样重要，都是为了防止误判：
      推理型模型常常不支持 stop、tools、logprobs —— 这是上游的声明，
      不是网关把功能砍了。当成「功能缺失」会得出错误结论。

    典型文案：
      Function tools with reasoning_effort are not supported for gpt-6-astra
      [invalid_params]: Unsupported parameter: 'stop' is not supported with this model
    """
    if not isinstance(err, ProbeError):
        return False
    if err.status not in (400, 422):
        return False
    text = (str(err) + " " + str(err.body or "")).lower()
    markers = (
        "not supported",
        "unsupported parameter",
        "unsupported_parameter",
        "does not support",
        "不支持",
    )
    return any(m in text for m in markers)


def _needs_bigger_max_tokens(err):
    """判断这个错误是不是「max_tokens 给得太小」。

    推理型模型收到过小的 max_tokens 会直接回 400，文案各家不同：
      OpenAI / 网关: Could not finish the message because max_tokens or model output limit was reached
      其它:          max_tokens too small / output limit
    共同点是状态码 400 且提到 max_tokens 或输出额度。
    """
    if not isinstance(err, ProbeError):
        return False
    if err.status != 400:
        return False
    text = (str(err) + " " + str(err.body or "")).lower()
    return ("max_tokens" in text) or ("output limit" in text) or ("输出" in text and "额度" in text)


def _reasoning_text(body):
    if not isinstance(body, dict):
        return ""
    try:
        msg = body["choices"][0].get("message") or {}
    except Exception:
        return ""
    if not isinstance(msg, dict):
        return ""
    for key in ("reasoning_content", "reasoning"):
        v = msg.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


# ---------------------------------------------------------------------------
# 适配层：OpenAI 兼容 / Anthropic / Gemini
# ---------------------------------------------------------------------------


class Adapter(object):
    name = "openai"

    def __init__(self, base_url, api_key, model, timeout=60.0, insecure=False):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout = timeout
        self.insecure = insecure
        self.last_headers = {}
        # 这个端点能接受的最小 max_tokens。推理型模型常常不能给 1，
        # 一旦试探成功就缓存，避免每次请求都先失败一轮。
        self._min_completion_tokens = 1
        # 可接受的 temperature。绝大多数端点接受 0，但有些模型（实测 kimi-k3.1-pro）
        # 只接受 1 或不传，传 0 会直接 400。一旦试探出可用值就锁定。
        self._preferred_temperature = 0.0
        self._temperature_locked = False
        self.temperature_forced = None   # 报告里要说明「确定性被牺牲了」

    # -- 子类实现 ----------------------------------------------------------
    def chat_url(self):
        raise NotImplementedError

    def auth_headers(self):
        return {}

    def build(self, messages, **opts):
        raise NotImplementedError

    def text_of(self, body):
        raise NotImplementedError

    def usage_of(self, body):
        raise NotImplementedError

    def model_of(self, body):
        raise NotImplementedError

    # -- 通用 --------------------------------------------------------------
    def ask(self, messages, **opts):
        """发一次非流式请求，返回 (text, usage_dict, body)。

        ★ 这里包了一层 temperature 兜底：有些模型只接受特定取值
          （实测 kimi-k3.1-pro 只认 1 或不传，传 0 直接 400），
          而绝大多数探测都要 temperature=0 才有确定性。
          所以先按期望值发，400 就换 1，再不行就不带这个参数，
          成功一次就锁定，后续请求不再重复失败。
        """
        want = opts.pop("temperature", self._preferred_temperature)
        attempts = []
        for t in (want, 1.0, None):
            if t not in attempts:
                attempts.append(t)

        # 已经锁定过就不用再试错
        if self._temperature_locked:
            attempts = [self._preferred_temperature]

        last = None
        for t in attempts:
            try:
                text, usage, body = self._ask_once(messages, temperature=t, **opts)
            except ProbeError as e:
                last = e
                # 只有 400 才怀疑是 temperature 的问题
                if e.status == 400 and not self._temperature_locked:
                    continue
                raise
            if t != want and not self._temperature_locked:
                self._preferred_temperature = t
                self._temperature_locked = True
                self.temperature_forced = t
            elif not self._temperature_locked:
                self._temperature_locked = True
                self._preferred_temperature = t
            return text, usage, body
        raise last if last else ProbeError("请求失败")

    def _ask_once(self, messages, **opts):
        payload = self.build(messages, **opts)
        status, body, raw = http_json(
            self.chat_url(),
            self.auth_headers(),
            payload,
            timeout=self.timeout,
            insecure=self.insecure,
        )
        if status != 200 or not isinstance(body, dict):
            err = ""
            if isinstance(body, dict):
                e = body.get("error")
                if isinstance(e, dict):
                    err = e.get("message") or json.dumps(e, ensure_ascii=False)[:200]
                elif e:
                    err = str(e)[:200]
            if not err and raw:
                err = raw[:200]
            raise ProbeError("HTTP %s %s" % (status, err), status, raw)
        return self.text_of(body), self.usage_of(body), body

    def ask_stream(self, messages, **opts):
        payload = self.build(messages, **opts)
        payload["stream"] = True
        # 流式请求也要用已验证过的 temperature，否则同样会 400
        if "temperature" in payload and self._temperature_locked:
            if self._preferred_temperature is None:
                payload.pop("temperature", None)
            else:
                payload["temperature"] = self._preferred_temperature
        elif "temperature" in payload and not self._temperature_locked:
            payload["temperature"] = self._preferred_temperature
        return http_stream_ttft(
            self.chat_url(),
            self.auth_headers(),
            payload,
            timeout=self.timeout,
            insecure=self.insecure,
        )

    def ask_text(self, messages, max_tokens=TEXT_MAX_TOKENS, **opts):
        """要正文的请求。正文为空且出现了推理内容时，自动加大预算重试一次。

        返回 (text, usage, body, grew)。grew 表示是否发生过重试，
        报告里会注明，免得用户以为是模型答不出来。
        """
        text, usage, body = self.ask(messages, max_tokens=max_tokens, **opts)
        if text.strip() or not _reasoning_text(body):
            return text, usage, body, False

        bigger = min(TEXT_MAX_TOKENS_CAP, max(TEXT_MAX_TOKENS, max_tokens * TEXT_MAX_TOKENS_GROWTH))
        if bigger <= max_tokens:
            return text, usage, body, False
        text2, usage2, body2 = self.ask(messages, max_tokens=bigger, **opts)
        if text2.strip():
            return text2, usage2, body2, True
        return text2 or text, usage2 or usage, body2 or body, True

    def prompt_tokens(self, text):
        """只让模型尽量少输出，把输入侧的账孤立出来。

        ★ 不能一律用 max_tokens=1。推理型模型（Claude thinking、o1/o3、
          gpt-6 这类）连「思考」都放不下，上游会直接回
          「Could not finish the message because max_tokens or model output
          limit was reached」—— 实测 gpt-6-astra 就要求至少 8。
          所以这里按 1 → 8 → 32 → 128 逐级试探，并把成功值缓存下来。
        """
        ladder = (1, 8, 32, 128)
        start = self._min_completion_tokens
        try:
            idx = ladder.index(start)
        except ValueError:
            idx = 0
        last = None
        for mt in ladder[idx:]:
            try:
                _, usage, _ = self.ask(
                    [{"role": "user", "content": text}], max_tokens=mt, temperature=0
                )
            except ProbeError as e:
                last = e
                if _needs_bigger_max_tokens(e):
                    continue
                raise
            if not isinstance(usage, dict):
                raise ProbeError("响应里没有 usage 字段")
            n = usage.get("prompt_tokens")
            if n is None:
                n = usage.get("input_tokens")
            if n is None:
                raise ProbeError("usage 里没有 prompt_tokens / input_tokens")
            self._min_completion_tokens = mt
            return int(n)
        raise last if last else ProbeError("prompt_tokens 全部失败")


class OpenAIAdapter(Adapter):
    """OpenAI 兼容接口。绝大多数中转站、DeepSeek、Qwen、Kimi、GLM、
    Groq、Together、OpenRouter 都走这一套。"""

    name = "openai"

    def chat_url(self):
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def auth_headers(self):
        return {"Authorization": "Bearer " + self.api_key}

    def build(self, messages, **opts):
        payload = {"model": self.model, "messages": messages}
        payload.update({k: v for k, v in opts.items() if v is not None})
        return payload

    def text_of(self, body):
        try:
            return body["choices"][0]["message"].get("content") or ""
        except Exception:
            return ""

    def usage_of(self, body):
        u = body.get("usage")
        return u if isinstance(u, dict) else None

    def model_of(self, body):
        return body.get("model")


class AnthropicAdapter(Adapter):
    """Anthropic 原生接口 /v1/messages。"""

    name = "anthropic"

    def chat_url(self):
        if self.base_url.endswith("/messages"):
            return self.base_url
        return self.base_url + "/messages"

    def auth_headers(self):
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    def build(self, messages, **opts):
        # Anthropic 把 system 单独拿出来
        system_parts = []
        rest = []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(m.get("content") or "")
            else:
                rest.append(m)
        payload = {
            "model": self.model,
            "messages": rest,
            "max_tokens": opts.get("max_tokens", 16),
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        if opts.get("temperature") is not None:
            payload["temperature"] = opts["temperature"]
        return payload

    def text_of(self, body):
        try:
            return "".join(
                b.get("text", "") for b in body.get("content", []) if isinstance(b, dict)
            )
        except Exception:
            return ""

    def usage_of(self, body):
        u = body.get("usage")
        if not isinstance(u, dict):
            return None
        out = dict(u)
        if "input_tokens" in u and "prompt_tokens" not in u:
            out["prompt_tokens"] = u["input_tokens"]
        if "output_tokens" in u and "completion_tokens" not in u:
            out["completion_tokens"] = u["output_tokens"]
        return out

    def model_of(self, body):
        return body.get("model")


class GeminiAdapter(Adapter):
    """Google Gemini generateContent 接口。"""

    name = "gemini"

    def chat_url(self):
        base = self.base_url
        if base.endswith(":generateContent"):
            return base
        if "/models/" in base:
            return base + ":generateContent"
        return base + "/models/" + self.model + ":generateContent"

    def auth_headers(self):
        return {"x-goog-api-key": self.api_key}

    def build(self, messages, **opts):
        contents = []
        system_parts = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                system_parts.append(m.get("content") or "")
                continue
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": m.get("content") or ""}],
                }
            )
        payload = {"contents": contents}
        gen = {}
        if opts.get("max_tokens") is not None:
            gen["maxOutputTokens"] = opts["max_tokens"]
        if opts.get("temperature") is not None:
            gen["temperature"] = opts["temperature"]
        if gen:
            payload["generationConfig"] = gen
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
        return payload

    def text_of(self, body):
        try:
            return "".join(
                p.get("text", "")
                for p in body["candidates"][0]["content"]["parts"]
                if isinstance(p, dict)
            )
        except Exception:
            return ""

    def usage_of(self, body):
        u = body.get("usageMetadata")
        if not isinstance(u, dict):
            return None
        return {
            "prompt_tokens": u.get("promptTokenCount"),
            "completion_tokens": u.get("candidatesTokenCount"),
            "total_tokens": u.get("totalTokenCount"),
        }

    def model_of(self, body):
        return body.get("modelVersion")


ADAPTERS = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
}


def make_adapter(kind, base_url, api_key, model, timeout=60.0, insecure=False):
    cls = ADAPTERS.get(kind)
    if not cls:
        raise SystemExit("未知的 --api 类型：%s（可选 openai / anthropic / gemini）" % kind)
    return cls(base_url, api_key, model, timeout=timeout, insecure=insecure)


# ---------------------------------------------------------------------------
# 探针
# ---------------------------------------------------------------------------


class Result(object):
    """一个探针的结果。

    verdict 取值：
      pass     通过（没发现异常）
      warn     有异常，需要人工确认
      fail     强证据表明不一致
      skip     无参照或不支持，不下结论
      error    执行失败（网络等）
    """

    def __init__(self, key, title, verdict, summary, detail=None, evidence=None):
        self.key = key
        self.title = title
        self.verdict = verdict
        self.summary = summary
        self.detail = detail or []
        self.evidence = evidence or {}


def probe_usage_consistency(ad, ctx):
    """用量自洽性与重复性。

    这是唯一一个完全不依赖「模型是什么」的检查：
    同一个输入，端点每次算出来的 prompt_tokens 必须一模一样。
    抖动说明计费链路本身有问题，这比模型真假更该先解决。
    """
    sample = "这是一段用来检查用量稳定性的中文文本，重复发送五次。" * 4
    seen = []
    errs = []
    for _ in range(5):
        try:
            seen.append(ad.prompt_tokens(sample))
        except ProbeError as e:
            errs.append(str(e))

    if len(seen) < 3:
        return Result(
            "usage",
            "用量稳定性",
            "error",
            "无法完成（%d/5 次成功）" % len(seen),
            ["最后一个错误：" + (errs[-1] if errs else "未知")],
        )

    uniq = sorted(set(seen))
    detail = ["同一段文本发送 5 次，prompt_tokens 取值：" + str(uniq)]

    if len(uniq) > 1:
        return Result(
            "usage",
            "用量稳定性",
            "fail",
            "同一输入算出 %d 种不同的 token 数" % len(uniq),
            detail
            + [
                "同一个输入不该出现不同的 prompt_tokens。",
                "这通常意味着计费链路不稳定，或者中间层在输入上做了不确定的改写。",
            ],
            {"values": uniq},
        )

    # 另测一次最小请求，得到「你什么都没发也要付的那部分」
    fixed_overhead = None
    try:
        fixed_overhead = ad.prompt_tokens("ping")
    except ProbeError:
        pass

    # 自洽性：总量是否等于分项之和
    try:
        _, usage, _ = ad.ask([{"role": "user", "content": "ping"}], max_tokens=8, temperature=0)
    except ProbeError:
        usage = None

    if isinstance(usage, dict):
        p = usage.get("prompt_tokens")
        c = usage.get("completion_tokens")
        t = usage.get("total_tokens")
        if all(isinstance(x, int) for x in (p, c, t)):
            detail.append("一次最小请求的 usage：prompt=%d completion=%d total=%d" % (p, c, t))
            if p + c != t:
                return Result(
                    "usage",
                    "用量稳定性",
                    "warn",
                    "total_tokens 不等于 prompt + completion（%d+%d≠%d）" % (p, c, t),
                    detail,
                    {"prompt": p, "completion": c, "total": t},
                )

    detail.append("测试样本本身：%d token" % seen[0])
    if fixed_overhead is not None:
        detail.append("固定开销（只发 'ping' 也要付的部分）：%d token" % fixed_overhead)
        if fixed_overhead > 40:
            detail.append(
                "★ 固定开销偏高。角色标记和模板一般是个位数到十几个，"
                "二三十往上值得问一句这部分钱花在哪了。"
            )
    detail.append("★ 边界：这里只证明「账算得稳」，不证明「账算得准」。")

    return Result(
        "usage", "用量稳定性", "pass", "5 次一致（样本 %d token）" % seen[0], detail, {"values": uniq}
    )


def measure_fingerprint(ad):
    """量一个端点的分词器指纹，返回 (deltas, baseline, selftest_ok, reasons)。

    这是 fingerprint 子命令的核心，也是「自己建基线」的基础。
    """
    try:
        base = ad.prompt_tokens("ok")
    except ProbeError as e:
        return None, None, False, ["基线请求失败：" + str(e)]

    deltas = {}
    reasons = []
    for key, text in FP_PROBES:
        try:
            deltas[key] = ad.prompt_tokens("ok" + text) - base
        except ProbeError as e:
            reasons.append("%s: %s" % (key, e))

    if len(deltas) < 3:
        return deltas, base, False, reasons + ["成功的探针太少，不足以构成指纹"]

    # 与 probe_tokenizer 相同的自检
    ratio = None
    if deltas.get("han_dup") and deltas.get("han_rare"):
        ratio = deltas["han_rare"] / float(deltas["han_dup"])
        if ratio < FP_RARE_RATIO_MIN:
            reasons.append(
                "han_rare/han_dup = %.2f < %.1f，不像真词表的输出"
                % (ratio, FP_RARE_RATIO_MIN)
            )
            return deltas, base, False, reasons
    return deltas, base, True, reasons


def cmd_fingerprint(args):
    """给一个端点建指纹基线，存成 JSON。

    用途：用官方渠道（或你信得过的渠道）跑一次，得到该模型的真实指纹；
    以后拿这个基线去验别的中转站给的是不是同一个模型。

        modelcheck fingerprint --base-url https://api.moonshot.cn/v1 \\
            --api-key sk-xxx --model kimi-k2 --out kimi-official.json

        modelcheck run --base-url https://some-relay/v1 \\
            --api-key sk-yyy --model kimi-k2 --profile kimi-official.json
    """
    api_key = args.api_key or os.environ.get("MODELCHECK_API_KEY", "")
    if not api_key and args.api != "gemini":
        print("缺少密钥：用 --api-key 或环境变量 MODELCHECK_API_KEY", file=sys.stderr)
        return 2

    ad = make_adapter(
        args.api, args.base_url, api_key, args.model,
        timeout=args.timeout, insecure=args.insecure,
    )
    deltas, base, ok, reasons = measure_fingerprint(ad)

    sys.stdout.write("\n" + _hr() + "\n")
    sys.stdout.write("  modelcheck %s   指纹基线\n" % __version__)
    sys.stdout.write("  端点 %s   模型 %s\n" % (args.base_url, args.model))
    sys.stdout.write(_hr() + "\n\n")

    if not deltas:
        sys.stdout.write("  X 一个探针都没成功，无法建基线\n")
        for r in reasons:
            sys.stdout.write("    - " + r + "\n")
        sys.stdout.write("\n")
        return 1

    sys.stdout.write("  基线固定开销：%s token\n\n" % base)
    sys.stdout.write("  %-10s %8s %12s\n" % ("探针", "增量", "每单位"))
    sys.stdout.write("  " + "-" * 34 + "\n")
    units = {"han_dup": 100, "han_rare": 120, "emoji": 60, "real_cn": 99, "mix": 90}
    for key, _ in FP_PROBES:
        if key not in deltas:
            continue
        u = units.get(key, 1)
        sys.stdout.write(
            "  %-10s %8d %12.3f\n" % (key, deltas[key], deltas[key] / float(u))
        )
    sys.stdout.write("\n")
    if ok:
        sys.stdout.write("  O 自检通过：这个指纹看起来是真词表的输出\n")
    else:
        sys.stdout.write("  ! 自检没通过，这份基线不可靠：\n")
        for r in reasons:
            sys.stdout.write("    - " + r + "\n")
    if ad.temperature_forced is not None:
        sys.stdout.write(
            "  ! 该端点只接受 temperature=%s。\n"
            "    指纹本身不受影响（prompt_tokens 只取决于输入怎么分词，和采样无关），\n"
            "    但用这个端点验「能力题」时，非零温度会让结果不稳定。\n"
            % repr(ad.temperature_forced)
        )

    profile = {
        "kind": "modelcheck-fingerprint",
        "version": __version__,
        "model": args.model,
        "api": args.api,
        "endpoint": args.base_url,
        "saved_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "baseline_tokens": base,
        "selftest_ok": ok,
        "deltas": deltas,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    sys.stdout.write("\n  已写入 %s\n" % args.out)
    sys.stdout.write(_hr() + "\n\n")
    sys.stdout.write(
        "  ★ 提醒：基线只对「同一模型」有效。拿它验同族不同型号（如 pro vs flash）\n"
        "    会得出错误的「不匹配」结论 —— 词表相同但指纹可能略有差异。\n\n"
    )
    return 0 if ok else 1


def load_profile(path):
    """读一份 fingerprint 基线，转成 VOCAB_REF 那种结构。"""
    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)
    if p.get("kind") != "modelcheck-fingerprint":
        raise SystemExit("不是 modelcheck 生成的基础文件：%s" % path)
    deltas = p.get("deltas") or {}
    fp = {k: v for k, v in deltas.items() if isinstance(v, int)}
    if not fp:
        raise SystemExit("基线文件里没有可用的指纹数据：%s" % path)
    name = "profile:" + (p.get("model") or os.path.basename(path))
    return name, {
        "label": "%s（来自 %s，%s）" % (p.get("model"), p.get("endpoint"), p.get("saved_at")),
        "confidence": "observed",
        "fp": fp,
        "source_model": p.get("model"),
    }


def build_refs(profile_paths, model):
    """合并内置参照和用户提供的基线。

    如果某个基线的 source_model 和当前模型完全一致，就把它当作「期望参照」，
    这意味着用户是拿官方基线来验第三方渠道 —— 这是这个工具最有价值的用法。
    """
    refs = dict(VOCAB_REF)
    expected = _expected_vocab_for(model)
    for path in profile_paths or []:
        name, info = load_profile(path)
        refs[name] = info
        src = (info.get("source_model") or "").lower()
        if src and model and src == model.lower():
            expected = name
    return refs, expected


def probe_tokenizer(ad, ctx):
    """分词器指纹：黑盒反推端点在用哪一套词表。

    原理：固定几段文本，用「加长后的 prompt_tokens 减去基线」得到增量。
    不同词表对同一文本的增量不同，而且比例差异明显（最强的一项差 1.57 倍）。

    先用 han_dup 自检：这段文本在 o200k 和 cl100k 下都应该是 50 token。
    如果实测偏离，说明 usage 不是后端词表的真实结果（可能是端点自己重算的），
    那么这个探针整体失效 —— 此时必须说「无法判断」，不能硬给结论。
    """
    refs = ctx.get("refs") or VOCAB_REF
    expected_vocab = ctx["expected_vocab"]
    if not expected_vocab:
        have = "、".join(sorted(refs))
        return Result(
            "tokenizer",
            "分词器指纹",
            "skip",
            "模型 %s 没有内置参照词表，跳过" % ctx["model"],
            [
                "内置参照：%s" % have,
                "",
                "其它模型族（Gemini / Llama / Qwen / Mistral 等）的分词器我没有实测值。",
                "与其编一个参照让它给出错误结论，不如明确跳过 ——",
                "这是这个工具「只做证伪」原则的一部分。",
            ],
        )

    try:
        base = ad.prompt_tokens("ok")
    except ProbeError as e:
        return Result("tokenizer", "分词器指纹", "error", "基线请求失败", [str(e)])

    deltas = {}
    errs = []
    for key, text in FP_PROBES:
        try:
            deltas[key] = ad.prompt_tokens("ok" + text) - base
        except ProbeError as e:
            errs.append("%s: %s" % (key, e))

    if len(deltas) < 3:
        return Result(
            "tokenizer",
            "分词器指纹",
            "error",
            "探针请求大部分失败（%d/%d）" % (len(deltas), len(FP_PROBES)),
            errs[:3],
        )

    detail = ["基线固定开销：%d token" % base, "", "探针增量（参照列是「声称的那个词表」应有的值）："]
    ref = refs[expected_vocab]["fp"]
    for key, _ in FP_PROBES:
        if key not in deltas:
            continue
        exp = ref.get(key)
        got = deltas[key]
        pct = (abs(got - exp) / exp * 100.0) if exp else 0.0
        detail.append("  %-9s 实测 %4d   参照 %4d   偏差 %5.1f%%" % (key, got, exp, pct))

    # ---- 自检：判断 usage 是不是后端分词器的真实结果 ----
    #
    # 两重检查，都和「具体是哪套词表」无关 —— 只回答「这像不像真 BPE 的输出」：
    #
    #  ① 罕见汉字必须比常用汉字贵。真实词表会把罕见字拆成好几个 token，
    #     han_rare / han_dup 落在 2.5~6.5 之间（OpenAI 系 5.2~6.0，Claude 约 2.6）。
    #
    #     按字符数拍脑袋的假分词器，这个比值恒等于 1.0 上下 —— 不管是「÷2」
    #     还是「÷1」还是「×1.2」，因为它对所有字符一视同仁。
    #     这是区分「真词表」和「假估算」最本质的一条。
    #
    #  ② 实测的 han_dup 必须能和某一套已知参照对上（±15%）。
    #     这条防的是「整体按比例缩放」的用法 —— 形状对，但绝对值被改过。
    problems = []
    ratio_rh = None
    if deltas.get("han_dup") and deltas.get("han_rare"):
        ratio_rh = deltas["han_rare"] / float(deltas["han_dup"])
        if ratio_rh < FP_RARE_RATIO_MIN:
            problems.append(
                "han_rare/han_dup = %.2f，真实词表应 ≥ %.1f"
                "（罕见汉字必须比常用汉字贵，这是 BPE 的本质）"
                % (ratio_rh, FP_RARE_RATIO_MIN)
            )

    known_handup = [refs[v]["fp"].get("han_dup") for v in refs]
    if deltas.get("han_dup") and known_handup:
        best_du = min(known_handup, key=lambda x: abs(x - deltas["han_dup"]))
        dev_du = abs(deltas["han_dup"] - best_du) / float(best_du)
        if dev_du > FP_HANDUP_TOL:
            problems.append(
                "han_dup 实测 %d，离最近的已知参照（%d）差 %.0f%%，"
                "说明用量被整体缩放或改写过"
                % (deltas["han_dup"], best_du, dev_du * 100)
            )

    if problems:
        detail += [
            "",
            "★ 自检没通过：",
        ]
        for p in problems:
            detail.append("    - " + p)
        detail += [
            "  这说明 usage 很可能是中间层自己算的，不是后端分词器的真实结果，",
            "  所以下面的指纹比对不可信。",
        ]
        return Result(
            "tokenizer",
            "分词器指纹",
            "skip",
            "usage 疑似被重算，指纹探针失效",
            detail
            + [
                "",
                "★ 这不是坏消息，也不是好消息 —— 是「这条线索断了」。",
                "  要判断模型真假，只能看功能矩阵和能力题那两项。",
            ],
            {"deltas": deltas, "selftest": False, "rare_ratio": ratio_rh},
        )

    # ---- 与两套参照比距离 ----
    scores = {}
    for vocab, info in refs.items():
        errs_ = []
        for key, exp in info["fp"].items():
            if key in deltas and exp:
                errs_.append(abs(deltas[key] - exp) / float(exp))
        if errs_:
            scores[vocab] = sum(errs_) / len(errs_)

    if not scores:
        return Result("tokenizer", "分词器指纹", "skip", "没有可比的参照项", detail)

    best = min(scores, key=lambda k: scores[k])
    best_err = scores[best]
    ranked = sorted(scores.items(), key=lambda kv: kv[1])

    detail += ["", "与参照的平均相对误差："]
    for vocab, e in ranked:
        mark = " ← 最接近" if vocab == best else ""
        detail.append("  %-13s %5.1f%%%s" % (vocab, e * 100, mark))

    ctx["observed_vocab"] = best if best_err <= FP_LOOSE_TOL else None
    ctx["observed_vocab_error"] = best_err

    if best_err <= FP_EXACT_TOL:
        if best == expected_vocab:
            conf = refs.get(best, {}).get("confidence", "measured")
            extra = []
            if conf == "observed":
                extra = [
                    "",
                    "★ 注意：这套参照是「实测观察值」，只有一处来源，没有和官方交叉验证。",
                    "  吻合说明「像」，不说明「一定是」。",
                ]
            return Result(
                "tokenizer",
                "分词器指纹",
                "pass",
                "与 %s 匹配（误差 %.1f%%）" % (best, best_err * 100),
                detail
                + extra
                + [
                    "",
                    "★ 边界：吻合只说明「词表像」，不说明「模型一定是」。",
                    "  同一个词表下有多个型号，这一项区分不了它们。",
                    "  要区分同族型号，看能力题和功能矩阵。",
                ],
                {"deltas": deltas, "scores": scores},
            )

        # 不匹配。结论有多硬，取决于「你声称的那套词表」的参照可信度：
        #   声称的是 measured（OpenAI 系） → 我们很清楚它该长什么样，不匹配就是强证据
        #   声称的是 observed（Claude）    → 参照本身只有一处来源，不匹配只能是弱信号
        expected_conf = refs.get(expected_vocab, {}).get("confidence", "measured")
        hard = expected_conf == "measured"
        return Result(
            "tokenizer",
            "分词器指纹",
            "fail" if hard else "warn",
            "词表是 %s，但声称的是 %s" % (best, expected_vocab),
            detail
            + [
                "",
                "★ %s"
                % (
                    "这是强信号：分词器是模型自带的，换模型一定会换词表。"
                    if hard
                    else "这是弱信号：声称的模型族只有实测观察参照，不足以定罪，建议结合能力题和趋势判断。"
                ),
                "  常见情况：声称高阶模型，实际走便宜的那一代。",
                "  再确认一遍：模型名有没有写错？有没有可能本来就标错了版本？",
            ],
            {"deltas": deltas, "scores": scores, "expected": expected_vocab, "observed": best},
        )

    if best_err <= FP_LOOSE_TOL:
        return Result(
            "tokenizer",
            "分词器指纹",
            "warn",
            "接近 %s（误差 %.1f%%），但不足以确认" % (best, best_err * 100),
            detail
            + [
                "",
                "★ 误差落在 8%~20% 之间：可能是端点在输入上做了轻微改写（拼接模板、",
                "  截断、转义），也可能是词表确实不同。建议换几段文本重测一次。",
            ],
            {"deltas": deltas, "scores": scores},
        )

    detail += [
        "",
        "★ 两套参照都对不上（最小误差 %.0f%%）。" % (best_err * 100),
        "  对声称 %s 的端点来说，这说明它用的不是 OpenAI 系词表 ——" % ctx["model"],
        "  但具体是哪一家的词表，我没有参照值，所以不下结论。",
    ]
    return Result(
        "tokenizer",
        "分词器指纹",
        "warn",
        "词表不属于任何已知的 OpenAI 参照",
        detail,
        {"deltas": deltas, "scores": scores},
    )


def probe_features(ad, ctx):
    """功能矩阵：声称的模型应该支持哪些接口能力。

    这一项是二值的、客观的，而且伪造成本不低 ——
    一个只会转发纯文本的套壳网关，很难凭空实现 logprobs、
    并行工具调用、严格 JSON 模式这些。
    """
    checks = []
    notes = []

    def add(name, ok, note="", na=False):
        checks.append({"name": name, "ok": ok, "note": note, "na": na})

    # 1) logprobs（OpenAI 系支持，很多套壳不支持）
    #
    #    ★ 用一道需要动脑的小题，而不是「Say hi」——
    #      这样一次请求能同时拿到两个信号：
    #        logprobs 有没有 → 网关有没有实现
    #        reasoning_content 有没有 → 这是不是推理型模型
    #      推理型模型通常不产出 logprobs，那是模型性质，不是功能被砍。
    try:
        _, _, body = ad.ask(
            [{"role": "user", "content": "一个正整数的平方是 729，这个数是多少？只输出数字。"}],
            max_tokens=TEXT_MAX_TOKENS,
            temperature=0,
            logprobs=True,
        )
        lp = None
        try:
            lp = body["choices"][0].get("logprobs")
        except Exception:
            pass
        is_reasoning = bool(_reasoning_text(body))
        if is_reasoning:
            ctx["saw_reasoning"] = True
        if lp:
            add("logprobs", True, "返回了 logprobs")
        elif is_reasoning:
            add(
                "logprobs",
                False,
                "不适用：这是推理型模型，不产出 logprobs",
                na=True,
            )
        else:
            add("logprobs", False, "参数被接受但没返回数据")
    except ProbeError as e:
        if _is_unsupported_param(e):
            add("logprobs", False, "不适用：上游声明该模型不支持 logprobs", na=True)
        else:
            add("logprobs", False, "不支持或报错：%s" % str(e)[:80])

    # 2) 严格 JSON 模式
    if ad.name == "openai":
        try:
            text, _, _, _ = ad.ask_text(
                [
                    {"role": "system", "content": "You output JSON only."},
                    {"role": "user", "content": 'Return exactly {"ok": true} and nothing else.'},
                ],
                max_tokens=60,
                temperature=0,
                response_format={"type": "json_object"},
            )
            parsed = None
            try:
                parsed = json.loads(text.strip())
            except Exception:
                m = re.search(r"\{.*\}", text, re.S)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except Exception:
                        parsed = None
            add(
                "response_format=json_object",
                isinstance(parsed, dict),
                "返回合法 JSON" if isinstance(parsed, dict) else "返回的不是合法 JSON：%r" % text[:60],
            )
        except ProbeError as e:
            add("response_format=json_object", False, "不支持或报错：%s" % str(e)[:80])

    # 3) 工具调用
    if ad.name == "openai":
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        try:
            _, _, body = ad.ask(
                [{"role": "user", "content": "What is the weather in Beijing? Use the tool."}],
                max_tokens=80,
                temperature=0,
                tools=tools,
                tool_choice="auto",
            )
            calls = None
            try:
                calls = body["choices"][0]["message"].get("tool_calls")
            except Exception:
                pass
            add(
                "function calling",
                bool(calls),
                "返回了 tool_calls" if calls else "没有返回 tool_calls",
            )
        except ProbeError as e:
            if _is_unsupported_param(e):
                add(
                    "function calling",
                    False,
                    "不适用：上游声明该模型不支持工具调用（%s）" % str(e)[:60],
                    na=True,
                )
            else:
                add("function calling", False, "不支持或报错：%s" % str(e)[:80])

    # 4) seed 参数（可复现性）
    if ad.name == "openai":
        try:
            t1, _, _, _ = ad.ask_text(
                [{"role": "user", "content": "从 1 数到 5，只输出数字，逗号分隔。"}],
                max_tokens=60,
                temperature=0,
                seed=42,
            )
            t2, _, _, _ = ad.ask_text(
                [{"role": "user", "content": "从 1 数到 5，只输出数字，逗号分隔。"}],
                max_tokens=60,
                temperature=0,
                seed=42,
            )
            same = t1.strip() == t2.strip()
            if same:
                add("seed 可复现", True, "两次输出一致")
            elif ctx.get("saw_reasoning"):
                # 推理型模型的思考过程本身是随机的，seed 管不住它，
                # 所以两次输出不同属正常，不该记成网关缺陷。
                add(
                    "seed 可复现",
                    False,
                    "不适用：推理型模型的思考过程不受 seed 约束",
                    na=True,
                )
            else:
                add("seed 可复现", False, "同 seed 两次输出不同")
        except ProbeError as e:
            add("seed 可复现", False, "不支持 seed：%s" % str(e)[:80])

    # 5) stop 序列是否生效
    try:
        text, _, _, _ = ad.ask_text(
            [{"role": "user", "content": "请连续输出字母 A，不要停。"}],
            max_tokens=60,
            temperature=0,
            stop=["AAAA"],
        )
        add("stop 生效", "AAAA" not in text, "输出里没有出现 stop 串" if "AAAA" not in text else "stop 串出现在输出里")
    except ProbeError as e:
        if _is_unsupported_param(e):
            add("stop 生效", False, "不适用：上游声明该模型不支持 stop 参数", na=True)
        else:
            add("stop 生效", False, "stop 参数报错：%s" % str(e)[:80])

    # 6) max_tokens 是否被尊重
    try:
        text, usage, _ = ad.ask(
            [{"role": "user", "content": "用一句话解释什么是 TCP 三次握手。"}],
            max_tokens=3,
            temperature=0,
        )
        c = (usage or {}).get("completion_tokens")
        if isinstance(c, int) and c > 12:
            add("max_tokens 被尊重", False, "要 3 个 token，实际生成 %d 个" % c)
        else:
            add("max_tokens 被尊重", True, "completion_tokens=%s" % c)
    except ProbeError as e:
        if _needs_bigger_max_tokens(e):
            # 推理型模型不接受这么小的额度，这项测不了，但不能算它失败
            add("max_tokens 被尊重", True, "不适用：该模型要求更大的输出额度，测不出上限")
        else:
            add("max_tokens 被尊重", False, "报错：%s" % str(e)[:80])

    # 7) 流式是否真流式
    try:
        first_ms, text, _, reasoning_blocks = ad.ask_stream(
            [{"role": "user", "content": "请写一段大约 150 字的话，介绍你自己。"}],
            max_tokens=TEXT_MAX_TOKENS,
            temperature=0.7,
        )
        got_any = bool(text) or reasoning_blocks > 0
        if got_any and not text and reasoning_blocks:
            detail_note = "只收到推理内容 %d 块，正文为空（模型在思考，不是没流式）" % reasoning_blocks
        elif got_any:
            detail_note = "首块 %.0f ms，累计 %d 字" % (first_ms, len(text))
        else:
            detail_note = "首块 %.0f ms，但一个字都没收到" % first_ms
        add("流式输出", got_any, detail_note)
        notes.append("流式首块延迟 %.0f ms（真流式通常 < 2000 ms）" % first_ms)
        ctx["ttft_ms"] = first_ms
        if reasoning_blocks:
            ctx["saw_reasoning"] = True
    except ProbeError as e:
        add("流式输出", False, "报错：%s" % str(e)[:80])

    # ★ 剔除「不适用」的项：既有按模型族静态剔除的，也有运行时由上游
    #   明确声明不支持的（na=True）。拿 OpenAI 的标准要求 Claude、
    #   或者要求推理型模型支持 stop，都会得出错误结论。
    na_map = _inapplicable_for(ctx.get("expected_vocab"))
    applicable = [
        c for c in checks if not c.get("na") and c["name"] not in na_map
    ]
    na_items = [c for c in checks if c.get("na") or c["name"] in na_map]

    passed = sum(1 for c in applicable if c["ok"])
    total = len(applicable)
    detail = []
    for c in applicable:
        detail.append("  %s %-28s %s" % ("O" if c["ok"] else "X", c["name"], c["note"]))
    for c in na_items:
        note = c["note"] if c.get("na") else ("不适用：" + na_map[c["name"]])
        detail.append("  - %-28s %s" % (c["name"], note))

    # 关键能力只在这些项适用时才作数。
    #
    # ★ logprobs 刻意不列入「关键」：推理型模型（o1/o3、gpt-6、Claude thinking）
    #   本来就不产出 logprobs，把它当硬指标会在正经推理模型上误报。
    #   「参数被接受但没返回数据」这件事本身仍然会写进明细，只是不吹哨。
    critical = ("function calling", "response_format=json_object")
    miss_critical = [
        c["name"] for c in applicable if c["name"] in critical and not c["ok"]
    ]

    detail.append("")
    detail.append("通过 %d/%d 项%s" % (passed, total, ("（另有 %d 项不适用）" % len(na_items)) if na_items else ""))
    for c in applicable:
        if c["name"] == "logprobs" and not c["ok"]:
            detail.append(
                "★ logprobs 缺失没有计入关键项：推理型模型本来就不产出它，"
                "单凭这一条不能说明网关砍了功能。"
            )
    detail.append("★ 边界：缺某项只能说明「这个网关没实现」，不能单独证明「模型被换了」。")
    detail.append("  但如果声称的是原厂高阶模型而关键能力缺失，解释成本很高。")

    if miss_critical:
        return Result(
            "features",
            "功能矩阵",
            "warn",
            "%d/%d 通过，缺失关键能力：%s" % (passed, total, "、".join(miss_critical)),
            detail,
            {"checks": checks, "passed": passed, "total": total},
        )
    if passed == total:
        return Result(
            "features", "功能矩阵", "pass", "%d/%d 全部通过" % (passed, total), detail, {"checks": checks}
        )
    return Result(
        "features",
        "功能矩阵",
        "pass",
        "%d/%d 通过，非关键项缺失" % (passed, total),
        detail,
        {"checks": checks},
    )


def _normalize_answer(text, kind):
    """从模型的自然语言回答里把答案抽出来。"""
    if not text:
        return ""
    t = text.strip()
    if kind == "int":
        nums = re.findall(r"-?\d[\d,]*", t)
        if not nums:
            return ""
        return nums[-1].replace(",", "")
    if kind == "char":
        m = re.search(r"\b([A-Za-z])\b", t)
        if m:
            return m.group(1).lower()
        for ch in t:
            if ch.isalpha():
                return ch.lower()
        return ""
    if kind == "seq":
        m = re.search(r"\b([A-Za-z]{2,6})\b", t)
        return m.group(1).upper() if m else ""
    return t.lower()


def probe_capability(ad, ctx):
    """能力题：一组答案唯一、可机器校验的题。

    这一项测的是「能力水平」，不是「模型身份」。
    通过率低说明后端不够强，但**不能**直接推出「被换了模型」——
    真模型也可能因为采样、截断、格式问题失手。
    """
    results = []
    correct = 0
    grew_any = False
    for item in CAPABILITY_QUESTIONS:
        try:
            text, _, body, grew = ad.ask_text(
                [{"role": "user", "content": item["q"]}],
                max_tokens=TEXT_MAX_TOKENS,
                temperature=0,
            )
            if grew:
                grew_any = True
            if not text.strip() and _reasoning_text(body):
                text = ""
        except ProbeError as e:
            results.append({"id": item["id"], "ok": None, "got": "", "err": str(e)[:100]})
            continue
        got = _normalize_answer(text, item["kind"])
        ok = got == item["expect"] or (item["kind"] == "seq" and got == item["expect"])
        if ok:
            correct += 1
        results.append(
            {"id": item["id"], "ok": ok, "got": got, "expect": item["expect"], "raw": text[:80]}
        )

    answered = [r for r in results if r["ok"] is not None]
    n = len(answered)
    if n == 0:
        return Result("capability", "能力基线", "error", "全部请求失败", ["见错误日志"])

    rate = correct / float(n)
    detail = []
    for r in results:
        if r["ok"] is None:
            detail.append("  ? %-8s 请求失败" % r["id"])
        elif r["ok"]:
            detail.append("  O %-8s 正确" % r["id"])
        else:
            detail.append(
                "  X %-8s 期望 %-6s 实际 %-10s 原文 %r"
                % (r["id"], r.get("expect", ""), r["got"], r.get("raw", "")[:50])
            )

    detail += [
        "",
        "通过 %d/%d = %.0f%%" % (correct, n, rate * 100),
    ]
    if grew_any:
        detail += [
            "★ 有题目第一次返回空正文、只有推理内容，已自动加大 max_tokens 重试。",
            "  这个模型的推理预算吃掉了默认额度 —— 这不是它答不出来，是预算给少了。",
        ]
    detail += [
        "★ 边界（这条最重要）：能力题测的是「水平」，不是「身份证」。",
        "  真模型也会因为采样、截断、指令理解差异失手，尤其是 temperature>0 时。",
        "  所以：通过率高**不能**证明是真模型；通过率低**也不能**直接定罪。",
        "  唯一可靠的用法是「纵向对比」——同一个端点每隔几天跑一次，看曲线有没有跳变。",
    ]

    if rate >= 0.8:
        return Result("capability", "能力基线", "pass", "%d/%d 通过（%.0f%%）" % (correct, n, rate * 100), detail, {"rate": rate, "results": results})
    if rate >= 0.5:
        return Result(
            "capability",
            "能力基线",
            "warn",
            "%d/%d 通过（%.0f%%），偏低" % (correct, n, rate * 100),
            detail,
            {"rate": rate, "results": results},
        )
    return Result(
        "capability",
        "能力基线",
        "warn",
        "%d/%d 通过（%.0f%%），明显偏低" % (correct, n, rate * 100),
        detail + ["", "★ 建议：先排除是否 max_tokens 截断、是否 temperature 过高，再怀疑模型。"],
        {"rate": rate, "results": results},
    )


def probe_context(ad, ctx):
    """上下文窗口实测：标称值和实际能到的经常不是一回事。

    做法是从小到大试探，找到第一次报错或明显截断的位置。
    为了省钱，用「重复词填充」而不是真实长文本，并且 max_tokens=1。

    ★ 默认只探到 32000 字符就停 —— 再往上输入 token 的费用会明显起来。
      要探到底加 --deep-context，自己清楚在为什么付钱。
    """
    filler = "测试内容 " * 40  # 约 180 字符
    ladder = [1000, 2000, 4000, 8000, 16000, 32000]
    if ctx.get("deep_context"):
        ladder += [64000, 128000]
    reached = 0
    last_err = ""
    detail = []
    for target_chars in ladder:
        reps = max(1, target_chars // len(filler))
        text = (filler * reps)[:target_chars]
        # 用 max_tokens=1 把成本压到最低，只测输入能不能进得去
        try:
            _, usage, _ = ad.ask(
                [{"role": "user", "content": text}], max_tokens=1, temperature=0
            )
            pt = (usage or {}).get("prompt_tokens")
            reached = target_chars
            detail.append("  %7d 字符  通过（prompt_tokens=%s）" % (target_chars, pt))
        except ProbeError as e:
            last_err = str(e)
            detail.append("  %7d 字符  失败：%s" % (target_chars, last_err[:90]))
            break

    if reached == 0:
        return Result(
            "context",
            "上下文窗口",
            "error",
            "一个长度都没通过，无法测",
            detail + ["最后一个错误：" + last_err],
        )

    detail += [
        "",
        "实测至少能接受 %d 字符的输入。" % reached,
        "★ 边界：这只测「输入进得去」，不测「进去之后还记不记得」。",
        "  真正的有效上下文要做「针在草堆里」测试（第 1 句埋一个事实，末尾提问），",
        "  那个成本高得多，本工具没做。",
    ]
    if reached <= 4000:
        return Result(
            "context",
            "上下文窗口",
            "warn",
            "只到 %d 字符就失败了" % reached,
            detail + ["", "★ 4000 字符以下就失败，和主流模型标称的 128k 差距很大。"],
            {"reached_chars": reached},
        )
    return Result(
        "context",
        "上下文窗口",
        "pass",
        "至少接受 %d 字符" % reached,
        detail,
        {"reached_chars": reached},
    )


def probe_errors(ad, ctx):
    """错误码规范性。

    这一项不判断模型真假，只判断这个网关好不好接 ——
    客户端重试逻辑依赖状态码区分，糊成一个码会导致重试策略写不对。
    """
    detail = []
    seen = {}

    # 不存在的模型
    old_model = ad.model
    ad.model = "no-such-model-xyz-000"
    try:
        ad.ask([{"role": "user", "content": "ping"}], max_tokens=4)
        seen["bad_model"] = ("200", "居然成功了")
    except ProbeError as e:
        seen["bad_model"] = (str(e.status) if e.status else "无状态码", str(e)[:80])
    finally:
        ad.model = old_model

    # 无效密钥（换个 adapter 实例，避免污染）
    bad = make_adapter(
        ad.name, ad.base_url, "sk-invalid-key-for-test", ctx["model"], timeout=20.0, insecure=ctx.get("insecure", False)
    )
    try:
        bad.ask([{"role": "user", "content": "ping"}], max_tokens=4)
        seen["bad_key"] = ("200", "居然成功了")
    except ProbeError as e:
        seen["bad_key"] = (str(e.status) if e.status else "无状态码", str(e)[:80])

    for k, label in (("bad_model", "不存在的模型"), ("bad_key", "无效密钥")):
        code, msg = seen[k]
        detail.append("  %-12s → %-8s %s" % (label, code, msg))

    detail += [
        "",
        "期望：不存在的模型 → 400 或 404；无效密钥 → 401。",
        "★ 边界：这一项测的是工程质量，和「有没有换模型」没有关系。",
        "  网关把这些糊成一个 500 或自定义码时，你的重试策略会写错 —— 但模型可能是真的。",
    ]

    ok = seen["bad_key"][0] == "401"
    if ok:
        return Result("errors", "错误码规范性", "pass", "密钥无效返回 401", detail, seen)
    return Result(
        "errors",
        "错误码规范性",
        "warn",
        "密钥无效返回 %s，不是 401" % seen["bad_key"][0],
        detail,
        seen,
    )


PROBES = [
    ("usage", probe_usage_consistency),
    ("tokenizer", probe_tokenizer),
    ("features", probe_features),
    ("capability", probe_capability),
    ("context", probe_context),
    ("errors", probe_errors),
]


# ---------------------------------------------------------------------------
# 评分与结论
# ---------------------------------------------------------------------------


def _expected_vocab_for(model):
    m = (model or "").lower()
    for key, vocab in sorted(MODEL_VOCAB_HINTS, key=lambda kv: -len(kv[0])):
        if key in m:
            return vocab
    return None


def score(results):
    """把各探针结果汇总成一个结论。

    权重刻意偏保守：
      - 分词器不匹配是最硬的证据（换模型必然换词表）
      - 功能缺失次之
      - 能力题最低（噪声最大，且只能看趋势）
    """
    by = {r.key: r for r in results}

    hard_fail = []
    warns = []
    skips = []

    tok = by.get("tokenizer")
    if tok:
        if tok.verdict == "fail":
            hard_fail.append("分词器指纹与声称的模型不符")
        elif tok.verdict == "warn":
            warns.append("分词器指纹有偏差")
        elif tok.verdict == "skip":
            skips.append("分词器指纹无法判定")

    feat = by.get("features")
    if feat:
        if feat.verdict == "warn":
            warns.append("功能矩阵有缺失")

    cap = by.get("capability")
    if cap and cap.verdict == "warn":
        warns.append("能力题通过率偏低")

    ctxr = by.get("context")
    if ctxr and ctxr.verdict == "warn":
        warns.append("上下文窗口明显偏小")

    usg = by.get("usage")
    if usg:
        if usg.verdict == "fail":
            hard_fail.append("同一输入的 token 数不稳定")
        elif usg.verdict == "warn":
            warns.append("usage 字段自相矛盾")
        elif usg.verdict == "error":
            skips.append("用量稳定性无法测")

    if hard_fail:
        verdict = "fail"
        headline = "高度疑似不一致"
    elif len(warns) >= 2:
        verdict = "warn"
        headline = "有异常，建议人工确认"
    elif len(warns) == 1:
        verdict = "warn"
        headline = "发现一项异常"
    elif skips:
        verdict = "partial"
        headline = "没发现异常，但有项目无法判定"
    else:
        verdict = "pass"
        headline = "未发现替换迹象"

    # 一个探针都没成功时，不能算「通过」——什么都测出来的结论没有意义
    usable = [r for r in results if r.verdict not in ("error",)]
    if not usable:
        verdict = "error"
        headline = "全部探针执行失败，什么都没测出来"

    return {
        "verdict": verdict,
        "headline": headline,
        "hard_fail": hard_fail,
        "warns": warns,
        "skips": skips,
    }


VERDICT_LABEL = {
    "pass": ("OK", "未发现替换迹象"),
    "partial": ("~", "部分项目无法判定"),
    "warn": ("!", "有异常，需要人工确认"),
    "fail": ("X", "高度疑似不一致"),
    "error": ("?", "全部探针执行失败"),
}


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _hr(ch="-", n=72):
    return ch * n


def render_console(report, unicode_ok=True):
    v = report["verdict"]
    label, desc = VERDICT_LABEL[v["verdict"]]
    lines = []
    lines.append("")
    lines.append(_hr())
    lines.append("  modelcheck %s   端点 %s" % (report["tool_version"], report["endpoint"]))
    lines.append("  模型 %s   接口 %s   时间 %s" % (report["model"], report["api"], report["time"]))
    lines.append(_hr())
    lines.append("")

    for r in report["results"]:
        mark, word = {
            "pass": ("+", "通过"),
            "warn": ("!", "注意"),
            "fail": ("X", "不通过"),
            "skip": ("-", "跳过"),
            "error": ("?", "失败"),
        }.get(r["verdict"], ("?", r["verdict"]))
        lines.append("  [%s] %-12s %s" % (mark, r["title"], r["summary"]))
        for d in r["detail"]:
            if d.strip():
                lines.append("      " + d)
        lines.append("")

    lines.append(_hr())
    lines.append("  结论：%s" % desc)
    for h in v["hard_fail"]:
        lines.append("    [X] " + h)
    for w in v["warns"]:
        lines.append("    [!] " + w)
    for s in v["skips"]:
        lines.append("    [-] " + s)
    lines.append(_hr())
    lines.append(
        "  ★ 这个工具只做证伪，不做证真。「未发现替换迹象」不等于「确认是真模型」。"
    )
    lines.append("    每个探针的边界都写在对应那一段下面，建议逐条读一遍再下结论。")
    lines.append("")
    return "\n".join(lines)


def render_markdown(report):
    v = report["verdict"]
    out = []
    out.append("# modelcheck 报告")
    out.append("")
    out.append("| 项 | 值 |")
    out.append("| --- | --- |")
    out.append("| 端点 | `%s` |" % report["endpoint"])
    out.append("| 模型 | `%s` |" % report["model"])
    out.append("| 接口类型 | %s |" % report["api"])
    out.append("| 时间 | %s |" % report["time"])
    out.append("| 工具版本 | %s |" % report["tool_version"])
    out.append("")
    out.append("## 结论：%s" % VERDICT_LABEL[v["verdict"]][1])
    out.append("")
    if v["hard_fail"]:
        out.append("**强信号：**")
        for h in v["hard_fail"]:
            out.append("- " + h)
        out.append("")
    if v["warns"]:
        out.append("**需要注意：**")
        for w in v["warns"]:
            out.append("- " + w)
        out.append("")
    if v["skips"]:
        out.append("**无法判定：**")
        for s in v["skips"]:
            out.append("- " + s)
        out.append("")
    out.append("## 各项明细")
    out.append("")
    for r in report["results"]:
        out.append("### %s — %s" % (r["title"], r["summary"]))
        out.append("")
        if r["detail"]:
            out.append("```")
            for d in r["detail"]:
                out.append(d)
            out.append("```")
            out.append("")
    out.append("---")
    out.append("")
    out.append("> 本工具只做证伪，不做证真。")
    out.append("> 「未发现替换迹象」不等于「确认真模型」。")
    out.append("")
    out.append("由上海治历科技有限公司（治历科技）维护 · https://github.com/zhilikj/modelcheck")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 基线：单次结果没有意义，趋势才有意义
# ---------------------------------------------------------------------------


def save_baseline(path, report):
    data = {
        "saved_at": report["time"],
        "model": report["model"],
        "api": report["api"],
        "results": [
            {"key": r["key"], "verdict": r["verdict"], "summary": r["summary"]}
            for r in report["results"]
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def compare_baseline(path, report):
    try:
        with open(path, "r", encoding="utf-8") as f:
            old = json.load(f)
    except Exception as e:
        return ["读取基线失败：" + str(e)]

    old_by = {r["key"]: r for r in old.get("results", [])}
    lines = []
    lines.append("基线时间：%s（模型 %s）" % (old.get("saved_at"), old.get("model")))
    lines.append("")
    lines.append("  %-12s %-10s %-10s %s" % ("项目", "基线", "现在", "变化"))
    changed = []
    for r in report["results"]:
        o = old_by.get(r["key"])
        if not o:
            lines.append("  %-12s %-10s %-10s %s" % (r["key"], "-", r["verdict"], "新增"))
            continue
        delta = ""
        if o["verdict"] != r["verdict"]:
            delta = "★ %s → %s" % (o["verdict"], r["verdict"])
            changed.append("%s: %s → %s" % (r["key"], o["verdict"], r["verdict"]))
        if o.get("summary") != r["summary"]:
            delta = (delta + "  " if delta else "") + "摘要变了"
        lines.append("  %-12s %-10s %-10s %s" % (r["key"], o["verdict"], r["verdict"], delta))
    lines.append("")
    if changed:
        lines.append("★ 有变化。单次结果不足以下结论，但同一条探针从 pass 变成 fail")
        lines.append("  值得立刻查一下是不是换了后端。")
    else:
        lines.append("与基线一致，没有变化。")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        prog="modelcheck",
        description="检测大模型 API 端点是否在掺水（换模型 / 降级 / 功能阉割）。只做证伪，不做证真。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例子：\n"
            "  # 验一个 OpenAI 兼容端点\n"
            "  python modelcheck.py run --base-url https://api.example.com/v1 \\\n"
            "      --api-key sk-xxx --model gpt-4o\n\n"
            "  # 验 Anthropic\n"
            "  python modelcheck.py run --api anthropic --base-url https://api.anthropic.com/v1 \\\n"
            "      --api-key sk-ant-xxx --model claude-sonnet-4-5\n\n"
            "  # 存基线，过几天再比\n"
            "  python modelcheck.py run ... --save-baseline base.json\n"
            "  python modelcheck.py run ... --compare-baseline base.json\n"
        ),
    )
    p.add_argument("--version", action="version", version="modelcheck " + __version__)

    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="跑一遍全部探针")
    r.add_argument("--base-url", required=True, help="接口地址，例如 https://api.openai.com/v1")
    r.add_argument("--api-key", default="", help="密钥（也可以用环境变量 MODELCHECK_API_KEY）")
    r.add_argument("--model", required=True, help="模型名，例如 gpt-4o")
    r.add_argument(
        "--api",
        default="openai",
        choices=sorted(ADAPTERS.keys()),
        help="接口类型，默认 openai（兼容绝大多数中转站）",
    )
    r.add_argument("--timeout", type=float, default=60.0, help="单请求超时秒数，默认 60")
    r.add_argument("--insecure", action="store_true", help="跳过 TLS 证书校验（自签名证书时用）")
    r.add_argument(
        "--deep-context",
        action="store_true",
        help="上下文探针一直探到 128000 字符（会产生较多输入 token 费用，默认到 32000 就停）",
    )
    r.add_argument("--only", default="", help="只跑指定探针，逗号分隔：" + ",".join(k for k, _ in PROBES))
    r.add_argument("--skip", default="", help="跳过指定探针，逗号分隔")
    r.add_argument("--json", dest="json_out", default="", help="把结果写到这个 JSON 文件")
    r.add_argument("--markdown", dest="md_out", default="", help="把报告写到这个 Markdown 文件")
    r.add_argument("--save-baseline", default="", help="把本次结果存成基线文件")
    r.add_argument("--compare-baseline", default="", help="和基线文件对比")
    r.add_argument(
        "--profile",
        action="append",
        default=[],
        metavar="FILE",
        help="加载一份 fingerprint 基线当参照（可重复）。"
        "若基线的模型名与 --model 一致，会作为「期望参照」——这是验第三方渠道的正确用法",
    )
    r.add_argument("--quiet", action="store_true", help="不打印明细，只打印结论")
    r.add_argument("--verbose", action="store_true", help="打印每个探针的执行进度")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("self-test", help="不联网的自检：验证参照数据和判定规则是否自洽")
    s.set_defaults(func=cmd_self_test)

    fp = sub.add_parser(
        "fingerprint",
        help="给一个端点建指纹基线（配合 run --profile 使用）",
    )
    fp.add_argument("--base-url", required=True)
    fp.add_argument("--api-key", default="")
    fp.add_argument("--model", required=True)
    fp.add_argument("--api", default="openai", choices=sorted(ADAPTERS.keys()))
    fp.add_argument("--timeout", type=float, default=60.0)
    fp.add_argument("--insecure", action="store_true")
    fp.add_argument("--out", required=True, help="基线写到这个 JSON 文件")
    fp.set_defaults(func=cmd_fingerprint)

    return p


def cmd_run(args):
    api_key = args.api_key or os.environ.get("MODELCHECK_API_KEY", "")
    if not api_key and args.api != "gemini":
        print("缺少密钥：用 --api-key 或环境变量 MODELCHECK_API_KEY", file=sys.stderr)
        return 2

    ad = make_adapter(
        args.api,
        args.base_url,
        api_key,
        args.model,
        timeout=args.timeout,
        insecure=args.insecure,
    )

    try:
        refs, expected_vocab = build_refs(args.profile, args.model)
    except SystemExit:
        raise
    except Exception as e:
        print("加载 --profile 失败：%s" % e, file=sys.stderr)
        return 2

    ctx = {
        "model": args.model,
        "api": args.api,
        "expected_vocab": expected_vocab,
        "refs": refs,
        "insecure": args.insecure,
        "deep_context": getattr(args, "deep_context", False),
    }

    only = set(x.strip() for x in args.only.split(",") if x.strip())
    skip = set(x.strip() for x in args.skip.split(",") if x.strip())

    results = []
    for key, fn in PROBES:
        if only and key not in only:
            continue
        if key in skip:
            continue
        if args.verbose:
            sys.stderr.write("  跑探针 %s ...\n" % key)
            sys.stderr.flush()
        try:
            results.append(fn(ad, ctx))
        except ProbeError as e:
            results.append(
                Result(key, key, "error", "探针执行失败", [str(e)[:300]])
            )
        except Exception as e:  # 探针本身有 bug 也不能整个崩掉
            results.append(
                Result(key, key, "error", "探针内部错误", ["%s: %s" % (type(e).__name__, e)])
            )

    report = {
        "tool": "modelcheck",
        "tool_version": __version__,
        "endpoint": args.base_url,
        "model": args.model,
        "api": args.api,
        "time": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "results": [
            {
                "key": r.key,
                "title": r.title,
                "verdict": r.verdict,
                "summary": r.summary,
                "detail": r.detail,
                "evidence": r.evidence,
            }
            for r in results
        ],
    }
    report["verdict"] = score(results)

    if not args.quiet:
        sys.stdout.write(render_console(report, _supports_unicode()))
    else:
        label, desc = VERDICT_LABEL[report["verdict"]["verdict"]]
        sys.stdout.write("结论：%s\n" % desc)

    if args.compare_baseline:
        lines = compare_baseline(args.compare_baseline, report)
        sys.stdout.write("\n" + _hr() + "\n  与基线对比\n" + _hr() + "\n")
        for l in lines:
            sys.stdout.write("  " + l + "\n")
        sys.stdout.write("\n")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        sys.stdout.write("  JSON 已写入 %s\n" % args.json_out)

    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write(render_markdown(report))
        sys.stdout.write("  Markdown 已写入 %s\n" % args.md_out)

    if args.save_baseline:
        save_baseline(args.save_baseline, report)
        sys.stdout.write("  基线已保存 %s\n" % args.save_baseline)

    verdict = report["verdict"]["verdict"]
    if verdict == "fail":
        return 1
    if verdict == "error":
        return 3
    return 0


def cmd_self_test(args):
    """不用密钥、不联网的自检：验证内置参照数据和判定逻辑本身是自洽的。

    它不测你的端点，只回答一个问题：「这个工具自己有没有毛病」。
    端到端的行为验证在 tests/ 目录里（用本地假端点）。
    """
    fails = []
    checks = 0

    def check(name, cond, note=""):
        nonlocal checks
        checks += 1
        print("  %s %-44s %s" % ("O" if cond else "X", name, note))
        if not cond:
            fails.append(name)

    print("")
    print("=" * 70)
    print("  modelcheck 自检   版本 %s" % __version__)
    print("=" * 70)
    print("")
    print("  [1] 参照数据自洽性")

    # 1) 每套参照都必须满足「罕见汉字比常用汉字贵」——真实 BPE 的共性
    for vocab in sorted(VOCAB_REF):
        fp = VOCAB_REF[vocab]["fp"]
        ratio = (fp["han_rare"] / float(fp["han_dup"])) if fp.get("han_dup") else 0
        check(
            "%s 的 han_rare/han_dup ≥ %.1f" % (vocab, FP_RARE_RATIO_MIN),
            ratio >= FP_RARE_RATIO_MIN,
            "%.2f" % ratio,
        )

    # 2) 判别用探针必须有明显区分度（跨任意两套参照）
    vocab_list = sorted(VOCAB_REF)
    for key in ("han_rare", "emoji", "real_cn", "mix", "han_dup"):
        vals = [VOCAB_REF[v]["fp"].get(key) for v in vocab_list]
        distinct = len(set(vals))
        check(
            "%s 在不同参照间有差异" % key,
            distinct >= 2,
            "取值 %s" % vals,
        )

    # 3) 置信度字段必须齐全且取值合法
    for vocab in vocab_list:
        conf = VOCAB_REF[vocab].get("confidence")
        check(
            "%s 声明了置信度" % vocab,
            conf in ("measured", "observed"),
            str(conf),
        )

    print("")
    print("  [2] 模型名 → 词表 映射")
    cases = [
        ("gpt-4o", "o200k_base"),
        ("gpt-4o-mini", "o200k_base"),
        ("gpt-4.1", "o200k_base"),
        ("gpt-5", "o200k_base"),
        ("o3-mini", "o200k_base"),
        ("gpt-4-turbo", "cl100k_base"),
        ("gpt-4", "cl100k_base"),
        ("gpt-3.5-turbo", "cl100k_base"),
    ]
    for model, want in cases:
        got = _expected_vocab_for(model)
        check("%-16s → %s" % (model, want), got == want, "实际 %s" % got)

    # gpt-4 不能被 gpt-4o 抢走（长串优先）
    check(
        "gpt-4 没有被 gpt-4o 抢走",
        _expected_vocab_for("gpt-4") == "cl100k_base",
        "gpt-4 → %s" % _expected_vocab_for("gpt-4"),
    )

    # 没有参照的模型要返回 None，而不是瞎猜
    for model in ("gemini-2.5-pro", "deepseek-chat", "qwen-max", "llama-3.1-70b"):
        check("%-20s → 无参照（跳过）" % model, _expected_vocab_for(model) is None)

    # Anthropic 系现在有参照了（observed 级）
    for model in ("claude-sonnet-4-5", "claude-3-5-haiku", "claude-opus-4"):
        got = _expected_vocab_for(model)
        check(
            "%-18s → anthropic_claude" % model,
            got == "anthropic_claude",
            "实际 %s" % got,
        )

    print("")
    print("  [3] 答案解析")
    cases = [
        ("33033", "int", "33033"),
        ("答案是 33033。", "int", "33033"),
        ("33,033", "int", "33033"),
        ("f", "char", "f"),
        ("字符是 f。", "char", "f"),
        ("BAC", "seq", "BAC"),
        ("顺序为 BAC", "seq", "BAC"),
    ]
    for raw, kind, want in cases:
        got = _normalize_answer(raw, kind)
        check("%-14r %-5s → %-6s" % (raw, kind, want), got == want, "实际 %r" % got)

    print("")
    print("  [4] 判定规则")

    def mk(key, verdict, summary=""):
        return Result(key, key, verdict, summary)

    scenarios = [
        ("全通过 → 未发现迹象", [mk("usage", "pass"), mk("tokenizer", "pass"), mk("features", "pass")], "pass"),
        ("分词器不符 → 高度疑似", [mk("tokenizer", "fail")], "fail"),
        ("两项异常 → 需人工确认", [mk("features", "warn"), mk("capability", "warn")], "warn"),
        ("一项异常 → 发现异常", [mk("context", "warn")], "warn"),
        ("全跳过 → 部分无法判定", [mk("tokenizer", "skip")], "partial"),
        ("用量不稳 → 高度疑似", [mk("usage", "fail")], "fail"),
    ]
    for name, results, want in scenarios:
        got = score(results)["verdict"]
        check(name, got == want, "实际 %s" % got)

    print("")
    print("  [5] 渲染不崩")
    demo = {
        "tool": "modelcheck",
        "tool_version": __version__,
        "endpoint": "https://example.com/v1",
        "model": "gpt-4o",
        "api": "openai",
        "time": "2026-01-01 00:00:00 +0800",
        "results": [
            {
                "key": "tokenizer",
                "title": "分词器指纹",
                "verdict": "pass",
                "summary": "与 o200k_base 匹配",
                "detail": ["一行明细"],
                "evidence": {},
            }
        ],
    }
    demo["verdict"] = {
        "verdict": "pass",
        "headline": "未发现替换迹象",
        "hard_fail": [],
        "warns": [],
        "skips": [],
    }
    try:
        a = render_console(demo, True)
        b = render_markdown(demo)
        check("控制台渲染", len(a) > 100, "%d 字符" % len(a))
        check("Markdown 渲染", len(b) > 100, "%d 字符" % len(b))
    except Exception as e:
        check("渲染不崩", False, "%s: %s" % (type(e).__name__, e))

    print("")
    print("=" * 70)
    if fails:
        print("  自检未通过：%d/%d 项失败" % (len(fails), checks))
        for f in fails:
            print("    X " + f)
        print("=" * 70)
        return 1
    print("  自检通过：%d/%d 项全部正常" % (checks, checks))
    print("=" * 70)
    print("")
    print("  这只证明工具本身没毛病，不是对你的端点的判断。")
    print("  要验端点，用：python modelcheck.py run --base-url ... --model ...")
    print("")
    return 0


def main(argv=None):
    _enable_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
