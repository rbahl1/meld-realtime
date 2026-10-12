"""reply bank and deterministic selector (design 1.9)."""

from pathlib import Path

from meld_rt.response.selector import Response, ResponseSelector, load_bank  # noqa: F401

BANK_PATH = Path(__file__).with_name("bank.yaml")
