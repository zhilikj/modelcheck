# -*- coding: utf-8 -*-
"""modelcheck 的端到端测试。

全部跑在本地假端点上，不联网、不需要任何密钥：

    python -m unittest discover -s tests -v
    # 或
    python -m pytest tests -q

重点不是「覆盖率」，而是验证一件事：
**同一条探针，对诚实的端点、降级端点、套壳端点、重算 usage 的端点，
给出的结论是否不同，而且方向正确。**
"""

import json
import math
import os
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import mock_endpoint  # noqa: E402
import modelcheck as mc  # noqa: E402


class MockServer(object):
    """起一个指定行为的假端点。"""

    def __init__(self, mode):
        self.mode = mode
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), mock_endpoint.make_handler(mode))
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.srv.shutdown()
        self.srv.server_close()

    @property
    def base_url(self):
        return "http://127.0.0.1:%d/v1" % self.port


def run_probes(mode, model="gpt-4o", only=None):
    """对某个模式的假端点跑一遍探针，返回 (report, results)。"""
    with MockServer(mode) as srv:
        ad = mc.make_adapter("openai", srv.base_url, "sk-test-key-123456", model, timeout=20.0)
        ctx = {
            "model": model,
            "api": "openai",
            "expected_vocab": mc._expected_vocab_for(model),
            "insecure": False,
            "deep_context": False,
        }
        results = []
        for key, fn in mc.PROBES:
            if only and key not in only:
                continue
            results.append(fn(ad, ctx))
    report = {
        "tool": "modelcheck",
        "tool_version": mc.__version__,
        "endpoint": srv.base_url,
        "model": model,
        "api": "openai",
        "time": "2026-01-01 00:00:00 +0800",
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
    report["verdict"] = mc.score(results)
    return report, {r.key: r for r in results}


class TestReferenceData(unittest.TestCase):
    """参照数据必须和假端点用的那份一致，否则测试就失去意义了。"""

    def test_probe_texts_match(self):
        self.assertEqual(
            [k for k, _ in mc.FP_PROBES],
            [k for k, _ in mock_endpoint.FP_PROBES],
        )
        self.assertEqual(
            [t for _, t in mc.FP_PROBES],
            [t for _, t in mock_endpoint.FP_PROBES],
        )

    def test_vocab_ref_match(self):
        for vocab in mock_endpoint.REF:
            self.assertIn(vocab, mc.VOCAB_REF, vocab)
            self.assertEqual(
                mc.VOCAB_REF[vocab]["fp"], mock_endpoint.REF[vocab], vocab
            )

    def test_every_reference_has_confidence(self):
        for vocab, info in mc.VOCAB_REF.items():
            self.assertIn(info.get("confidence"), ("measured", "observed"), vocab)

    def test_rare_chars_cost_more_than_common(self):
        # 真实 BPE 的共性：罕见汉字的单价必须显著高于常用汉字。
        # 这条一旦不成立，自检就失去了区分「真词表」和「假估算」的能力。
        for vocab, info in mc.VOCAB_REF.items():
            fp = info["fp"]
            ratio = fp["han_rare"] / float(fp["han_dup"])
            self.assertGreaterEqual(ratio, mc.FP_RARE_RATIO_MIN, vocab)

    def test_discriminating_probes_actually_differ(self):
        for key in ("han_rare", "emoji", "real_cn", "mix", "han_dup"):
            vals = {mc.VOCAB_REF[v]["fp"][key] for v in mc.VOCAB_REF}
            self.assertGreaterEqual(len(vals), 2, "%s 在所有参照下相同，没有判别力" % key)

    def test_capability_answers(self):
        """题库的期望答案必须真的对。这条最容易写错。"""
        # 847 * 39
        self.assertEqual(847 * 39, 33033)
        # "abcdefgh" 反转后第 3 个字符
        self.assertEqual("abcdefgh"[::-1][2], "f")
        # 二进制 101101
        self.assertEqual(int("101101", 2), 45)
        # 3 步向 + 2 步下 = C(5,2)
        self.assertEqual(math.comb(5, 2), 10)
        # f(x)=3x+2, f(f(1))
        f = lambda x: 3 * x + 2
        self.assertEqual(f(f(1)), 17)
        # banana 里 a 的个数
        self.assertEqual("banana".count("a"), 3)
        # sum([1,2,3,4,5][1::2])
        self.assertEqual(sum([1, 2, 3, 4, 5][1::2]), 6)
        # 2,3,5,7,11,13,17 第 5 个
        self.assertEqual([2, 3, 5, 7, 11, 13, 17][4], 11)
        # 3*4*5
        self.assertEqual(3 * 4 * 5, 60)

        expected = {q["id"]: q["expect"] for q in mc.CAPABILITY_QUESTIONS}
        self.assertEqual(expected["mul"], "33033")
        self.assertEqual(expected["rev"], "f")
        self.assertEqual(expected["bin"], "45")
        self.assertEqual(expected["comp"], "10")
        self.assertEqual(expected["func"], "17")
        self.assertEqual(expected["count"], "3")
        self.assertEqual(expected["code"], "6")
        self.assertEqual(expected["logic"], "BAC")
        self.assertEqual(expected["prime"], "11")
        self.assertEqual(expected["units"], "60")


class TestHonestEndpoint(unittest.TestCase):
    """诚实的端点：不该被误判。"""

    @classmethod
    def setUpClass(cls):
        cls.report, cls.by = run_probes("real")

    def test_tokenizer_passes(self):
        r = self.by["tokenizer"]
        self.assertEqual(r.verdict, "pass", r.summary)
        self.assertIn("o200k_base", r.summary)

    def test_no_hard_fail(self):
        self.assertEqual(self.report["verdict"]["hard_fail"], [])

    def test_verdict_not_fail(self):
        self.assertNotEqual(self.report["verdict"]["verdict"], "fail")

    def test_capability_high(self):
        # 假端点在 real 模式下全答对
        self.assertGreaterEqual(self.by["capability"].evidence["rate"], 0.9)


class TestDowngradedEndpoint(unittest.TestCase):
    """降级端点：声称 gpt-4o，实际 cl100k_base。这是最该被抓的一类。"""

    @classmethod
    def setUpClass(cls):
        cls.report, cls.by = run_probes("downgrade")

    def test_tokenizer_fails(self):
        r = self.by["tokenizer"]
        self.assertEqual(r.verdict, "fail", r.summary)
        self.assertIn("cl100k_base", r.summary)

    def test_verdict_is_fail(self):
        self.assertEqual(self.report["verdict"]["verdict"], "fail")

    def test_hard_fail_mentions_vocab(self):
        joined = " ".join(self.report["verdict"]["hard_fail"])
        self.assertIn("分词器", joined)


class TestRecomputedUsage(unittest.TestCase):
    """usage 被端点自己重算：指纹探针必须自检失败，然后说「无法判断」。

    这一类最关键 —— 一个不成熟的工具会在这里给出一个假结论。
    """

    @classmethod
    def setUpClass(cls):
        cls.report, cls.by = run_probes("recomputed")

    def test_tokenizer_skips(self):
        r = self.by["tokenizer"]
        self.assertEqual(r.verdict, "skip", r.summary)
        self.assertIn("重算", r.summary)

    def test_does_not_claim_fail(self):
        # 不能因为测不出来就定罪
        self.assertNotIn("分词器", " ".join(self.report["verdict"]["hard_fail"]))

    def test_verdict_is_partial_or_warn(self):
        self.assertIn(self.report["verdict"]["verdict"], ("partial", "warn"))


class TestFakeEndpoint(unittest.TestCase):
    """套壳端点：词表对不上、功能缺失、能力题答错。"""

    @classmethod
    def setUpClass(cls):
        cls.report, cls.by = run_probes("fake")

    def test_tokenizer_not_pass(self):
        self.assertNotEqual(self.by["tokenizer"].verdict, "pass")

    def test_capability_low(self):
        self.assertLess(self.by["capability"].evidence["rate"], 0.9)

    def test_features_warn(self):
        self.assertEqual(self.by["features"].verdict, "warn")

    def test_verdict_is_warn_or_fail(self):
        self.assertIn(self.report["verdict"]["verdict"], ("warn", "fail"))


class TestUnknownModelSkipsFingerprint(unittest.TestCase):
    """没有参照的模型，指纹探针要跳过，而不是瞎猜。"""

    def test_gemini_skips(self):
        report, by = run_probes("real", model="gemini-2.5-pro", only=["tokenizer"])
        self.assertEqual(by["tokenizer"].verdict, "skip")
        self.assertIn("没有内置参照", by["tokenizer"].summary)


class TestClaudeReference(unittest.TestCase):
    """Claude 参照（observed 级）：吻合时说 pass，不吻合时只能说 warn。"""

    def test_claude_endpoint_matches(self):
        report, by = run_probes("claude", model="claude-fable-5-2", only=["tokenizer"])
        self.assertEqual(by["tokenizer"].verdict, "pass", by["tokenizer"].summary)
        self.assertIn("anthropic_claude", by["tokenizer"].summary)

    def test_claude_claim_but_openai_vocab_is_soft_fail(self):
        # 声称 Claude，实测是 o200k。因为 Claude 参照是 observed 级，
        # 不能像「声称 gpt-4o 却是 cl100k」那样直接判定 fail。
        report, by = run_probes("real", model="claude-fable-5-2", only=["tokenizer"])
        self.assertEqual(by["tokenizer"].verdict, "warn", by["tokenizer"].summary)
        self.assertIn("弱信号", " ".join(by["tokenizer"].detail))

    def test_claude_is_not_caught_by_selftest(self):
        # 关键回归：han_dup=100 的端点不能被误判为「usage 被重算」
        report, by = run_probes("claude", model="claude-fable-5-2", only=["tokenizer"])
        self.assertNotIn("重算", by["tokenizer"].summary)


class TestGlmReference(unittest.TestCase):
    """GLM 参照（observed 级）。GLM 的指纹和三家 OpenAI/Claude 都不一样，
    是第四套独立的词表 —— 这组测试防止它被误判成别的族。"""

    def test_glm_endpoint_matches(self):
        report, by = run_probes("glm", model="glm-5.3-flash", only=["tokenizer"])
        self.assertEqual(by["tokenizer"].verdict, "pass", by["tokenizer"].summary)
        self.assertIn("glm_zhipu", by["tokenizer"].summary)

    def test_glm_is_distinguishable_from_openai(self):
        # GLM 声称 + o200k 实测 → 弱信号（observed 参照不判 fail）
        report, by = run_probes("real", model="glm-5.3-flash", only=["tokenizer"])
        self.assertEqual(by["tokenizer"].verdict, "warn", by["tokenizer"].summary)

    def test_glm_reference_differs_from_all_others(self):
        glm = mc.VOCAB_REF["glm_zhipu"]["fp"]
        for other in ("o200k_base", "cl100k_base", "anthropic_claude"):
            ref = mc.VOCAB_REF[other]["fp"]
            errs = [abs(glm[k] - ref[k]) / float(ref[k]) for k in ref]
            avg = sum(errs) / len(errs)
            self.assertGreater(avg, 0.15, "GLM 和 %s 太像，判别力不足" % other)


class TestCli(unittest.TestCase):
    def test_run_writes_json_and_markdown(self):
        import tempfile

        with MockServer("real") as srv:
            tmpdir = tempfile.mkdtemp()
            jp = os.path.join(tmpdir, "r.json")
            mp = os.path.join(tmpdir, "r.md")
            bp = os.path.join(tmpdir, "base.json")
            rc = mc.main(
                [
                    "run",
                    "--base-url",
                    srv.base_url,
                    "--api-key",
                    "sk-test-key-123456",
                    "--model",
                    "gpt-4o",
                    "--only",
                    "usage,tokenizer",
                    "--json",
                    jp,
                    "--markdown",
                    mp,
                    "--save-baseline",
                    bp,
                    "--quiet",
                ]
            )
            self.assertIn(rc, (0, 1))
            self.assertTrue(os.path.exists(jp))
            self.assertTrue(os.path.exists(mp))
            self.assertTrue(os.path.exists(bp))

            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["model"], "gpt-4o")
            self.assertIn("verdict", data)

            with open(mp, encoding="utf-8") as f:
                md = f.read()
            self.assertIn("modelcheck 报告", md)

    def test_baseline_compare_no_change(self):
        import tempfile

        with MockServer("real") as srv:
            tmpdir = tempfile.mkdtemp()
            bp = os.path.join(tmpdir, "base.json")
            common = [
                "--base-url",
                srv.base_url,
                "--api-key",
                "sk-test-key-123456",
                "--model",
                "gpt-4o",
                "--only",
                "usage",
                "--quiet",
            ]
            mc.main(["run"] + common + ["--save-baseline", bp])
            # 再跑一次并对比
            rc = mc.main(["run"] + common + ["--compare-baseline", bp])
            self.assertIn(rc, (0, 1))


class TestSelfTest(unittest.TestCase):
    def test_self_test_passes(self):
        rc = mc.cmd_self_test(None)
        self.assertEqual(rc, 0)


class TestTemperatureFallback(unittest.TestCase):
    """有些模型只接受特定 temperature（实测 kimi-k3.1-pro 只认 1 或不传）。

    工具默认用 temperature=0 保证确定性，遇到这种端点必须能自动降级，
    否则所有探针都会 400 —— 整个工具在这个模型上就废了。
    """

    def test_probes_survive_strict_temperature(self):
        report, by = run_probes("strict_temp", model="gpt-4o", only=["usage", "tokenizer"])
        self.assertEqual(by["usage"].verdict, "pass", by["usage"].summary)
        self.assertEqual(by["tokenizer"].verdict, "pass", by["tokenizer"].summary)
        self.assertNotIn("重算", by["tokenizer"].summary)

    def test_temperature_forced_is_recorded(self):
        with MockServer("strict_temp") as srv:
            ad = mc.make_adapter(
                "openai", srv.base_url, "sk-test-key-123456", "gpt-4o", timeout=20.0
            )
            ad.prompt_tokens("ok")
            # 0 会被拒，工具应回落到 1 并记下来
            self.assertEqual(ad.temperature_forced, 1.0)


class TestFingerprintProfile(unittest.TestCase):
    """自己建基线 → 拿基线验别的渠道。这是工具最有价值的用法。"""

    def _make_profile(self, mode="real", model="gpt-4o"):
        import tempfile

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "p.json")
        with MockServer(mode) as srv:
            rc = mc.main(
                [
                    "fingerprint",
                    "--base-url",
                    srv.base_url,
                    "--api-key",
                    "sk-test-key-123456",
                    "--model",
                    model,
                    "--out",
                    path,
                ]
            )
        return rc, path

    def test_fingerprint_writes_file(self):
        rc, path = self._make_profile()
        self.assertEqual(rc, 0)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["kind"], "modelcheck-fingerprint")
        self.assertTrue(data["selftest_ok"])
        self.assertEqual(data["model"], "gpt-4o")
        self.assertIn("han_dup", data["deltas"])

    def test_profile_matches_same_model(self):
        rc, path = self._make_profile()
        with MockServer("real") as srv:
            refs, expected = mc.build_refs([path], "gpt-4o")
        self.assertTrue(expected.startswith("profile:"), expected)
        self.assertIn(expected, refs)

    def test_profile_does_not_hijack_other_models(self):
        rc, path = self._make_profile(model="gpt-4o")
        # 拿 gpt-4o 的基线去验 gpt-6，不应该抢走「期望参照」
        refs, expected = mc.build_refs([path], "gpt-6-astra")
        self.assertEqual(expected, "o200k_base")
        # 但基线仍然参与比较，只是不当期望值
        self.assertTrue(any(k.startswith("profile:") for k in refs))

    def test_bad_profile_rejected(self):
        import tempfile

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": "something-else"}, f)
        with self.assertRaises(SystemExit):
            mc.load_profile(path)


class TestUnreachableEndpoint(unittest.TestCase):
    """端点完全连不上时，不能输出「通过」—— 什么都没测出来的结论没有意义。"""

    def test_all_error_verdict(self):
        results = [mc.Result(k, k, "error", "连不上") for k, _ in mc.PROBES]
        self.assertEqual(mc.score(results)["verdict"], "error")

    def test_exit_code_is_3(self):
        # 端口 9 是 discard 服务，本机通常没有监听
        rc = mc.main(
            [
                "run",
                "--base-url",
                "http://127.0.0.1:9/v1",
                "--api-key",
                "sk-test-123456",
                "--model",
                "gpt-4o",
                "--only",
                "usage",
                "--quiet",
            ]
        )
        self.assertEqual(rc, 3)

    def test_partial_error_is_not_error_verdict(self):
        # 有任何一个探针成功，就不算「全失败」
        results = [
            mc.Result("usage", "usage", "pass", "OK"),
            mc.Result("tokenizer", "tokenizer", "error", "挂了"),
        ]
        self.assertEqual(mc.score(results)["verdict"], "pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
