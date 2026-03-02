import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import registry_pb2
from network.registry_server import (
    MAX_REPORTED_INPUT_TOKENS,
    MAX_REPORTED_OUTPUT_TOKENS,
    RegistryServicer,
)


class _Ctx:
    def __init__(self):
        self.code = None
        self.details = ""

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def test_report_usage_rejects_negative_tokens():
    svc = RegistryServicer(no_chain=True)
    ctx = _Ctx()
    resp = svc.ReportUsage(
        registry_pb2.ReportUsageRequest(
            input_tokens=-1, output_tokens=10, model_id="m"
        ),
        ctx,
    )
    assert resp.accepted is False
    assert resp.cost == 0.0
    assert "non-negative" in ctx.details


def test_report_usage_rejects_unbounded_input():
    svc = RegistryServicer(no_chain=True)
    ctx = _Ctx()
    resp = svc.ReportUsage(
        registry_pb2.ReportUsageRequest(
            input_tokens=MAX_REPORTED_INPUT_TOKENS + 1,
            output_tokens=10,
            model_id="m",
        ),
        ctx,
    )
    assert resp.accepted is False
    assert "input_tokens exceeds limit" in ctx.details


def test_report_usage_rejects_unbounded_output():
    svc = RegistryServicer(no_chain=True)
    ctx = _Ctx()
    resp = svc.ReportUsage(
        registry_pb2.ReportUsageRequest(
            input_tokens=10,
            output_tokens=MAX_REPORTED_OUTPUT_TOKENS + 1,
            model_id="m",
        ),
        ctx,
    )
    assert resp.accepted is False
    assert "output_tokens exceeds limit" in ctx.details


def test_report_usage_accepts_valid_counts():
    svc = RegistryServicer(no_chain=True)
    ctx = _Ctx()
    resp = svc.ReportUsage(
        registry_pb2.ReportUsageRequest(
            input_tokens=100,
            output_tokens=25,
            model_id="m",
        ),
        ctx,
    )
    assert resp.accepted is True
    assert resp.cost > 0
