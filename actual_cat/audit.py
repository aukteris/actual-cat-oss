import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    def __init__(self, log_path: str) -> None:
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _write(self, record: dict[str, Any]) -> None:
        record["run_id"] = self.run_id
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log(
        self,
        txn: Any,
        llm_response: dict[str, Any],
        mode: str,
        action: str,
        pipeline: str = "categorization",
        extra: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "event": "decision",
            "pipeline": pipeline,
            "mode": mode,
            "action": action,
            "transaction_id": txn.id,
            "transaction_date": str(txn.get_date()),
            "transaction_amount_cents": txn.amount,
            "transaction_account": txn.account.name if txn.account else None,
            "transaction_payee_imported": txn.imported_description,
            "transaction_payee": txn.payee.name if txn.payee else None,
            "llm_response": llm_response,
        }
        if extra:
            record.update(extra)
        self._write(record)

    def log_failure(self, txn: Any, error: str, pipeline: str) -> None:
        self._write({
            "event": "failure",
            "pipeline": pipeline,
            "transaction_id": txn.id,
            "error": error,
        })

    def log_skipped_pipeline(self, pipeline: str) -> None:
        self._write({"event": "pipeline_skipped", "pipeline": pipeline})

    def log_run_complete(self) -> None:
        self._write({"event": "run_complete"})
