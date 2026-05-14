from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.services.external_data_service import ExternalDataService  # noqa: E402


class ExternalDataServiceTests(unittest.TestCase):
    def test_cern_uses_cached_sample_when_live_url_is_unset(self) -> None:
        service = ExternalDataService()
        original_live_url = settings.cern_live_url
        settings.cern_live_url = ""
        try:
            event = asyncio.run(service._fetch_cern())
        finally:
            settings.cern_live_url = original_live_url

        self.assertEqual(event.source_name, "cern")
        self.assertEqual(event.event_type, "custom_json_feed")
        self.assertEqual(event.title, "Cryogenic Stability Event")
        self.assertEqual(event.payload["facility"], "CERN")


if __name__ == "__main__":
    unittest.main()
