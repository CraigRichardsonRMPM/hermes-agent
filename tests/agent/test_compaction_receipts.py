from __future__ import annotations

from agent.conversation_compression import (
    COMPACTION_RECEIPTS_MODEL_CONFIG_KEY,
    _build_compaction_receipt,
    _next_compaction_receipts,
)


class _ReceiptDB:
    def __init__(self, stored):
        self.stored = stored

    def get_session_model_config_value(self, session_id, key, default=None):
        assert session_id == "session-1"
        assert key == COMPACTION_RECEIPTS_MODEL_CONFIG_KEY
        return self.stored


class _Agent:
    session_id = "session-1"

    def __init__(self, stored=None):
        self._session_db = _ReceiptDB(stored)


def test_receipt_records_budget_boundary_and_preservation_without_message_content():
    before = [
        {"role": "system", "content": "secret system", "_row_id": 10},
        {"role": "user", "content": "critical task identity", "_row_id": 11},
        {"role": "assistant", "content": "old work", "_row_id": 12},
        {"role": "user", "content": "latest request", "_row_id": 13},
    ]
    after = [
        {"role": "system", "content": "secret system", "_row_id": 10},
        {"role": "assistant", "content": "generated summary"},
        {"role": "user", "content": "latest request", "_row_id": 13},
    ]

    receipt = _build_compaction_receipt(
        attempt_id="attempt-1",
        trigger_source="overflow",
        pre_messages=before,
        post_messages=after,
        compression_count=2,
        tail_messages=before[-1:],
        compressed_user_turn_outcome="already_present",
        reinserted_last_assistant_reply=False,
        timestamp="2026-10-09T12:00:00Z",
    )

    assert receipt["timestamp"] == "2026-10-09T12:00:00Z"
    assert receipt["trigger"] == "budget"
    assert receipt["trigger_source"] == "overflow"
    assert receipt["pre_compaction_head_ref"] == "row:10"
    assert receipt["preserved"] == {
        "verbatim_message_refs": ["row:10", "row:13"],
        "verbatim_message_count": 2,
        "generated_or_rewritten_message_count": 1,
        "verbatim_tail_count": 1,
        "real_user_turn": "already_present",
        "last_assistant_reply_reinserted": False,
    }
    assert "secret system" not in str(receipt)
    assert "critical task identity" not in str(receipt)


def test_manual_receipt_is_policy_and_history_is_bounded_for_successor_reads():
    receipt = _build_compaction_receipt(
        attempt_id="new",
        trigger_source="manual",
        pre_messages=[{"role": "user", "content": "x"}],
        post_messages=[{"role": "user", "content": "x"}],
        compression_count=25,
        tail_messages=None,
        compressed_user_turn_outcome="already_present",
        reinserted_last_assistant_reply=False,
        timestamp="2026-10-09T12:00:00Z",
    )
    stored = [{"attempt_id": f"old-{index}"} for index in range(25)]

    receipts = _next_compaction_receipts(_Agent(stored), receipt)

    assert len(receipts) == 20
    assert receipts[0]["attempt_id"] == "old-6"
    assert receipts[-1]["attempt_id"] == "new"
    assert receipts[-1]["trigger"] == "policy"


def test_malformed_stored_receipts_fail_closed_to_new_receipt():
    receipt = {"attempt_id": "new"}

    assert _next_compaction_receipts(_Agent({"not": "a-list"}), receipt) == [receipt]
