"""Demo controls for the dashboard: inject failures on purpose to watch the pipeline handle them.

Only wired up by ``python -m pipeline local``. A normal ``api`` deployment never
exposes these endpoints.
"""

from __future__ import annotations

import threading

from pipeline.classifier.model import IntentModel, Prediction


class DemoControls:
    """Shared switches between the API (which flips them) and the workers (which obey them)."""

    def __init__(self) -> None:
        self.model_outage = threading.Event()


class OutageAwareModel:
    """Wraps the real model. While an outage is on, every prediction raises, so messages retry with backoff."""

    def __init__(self, model: IntentModel, controls: DemoControls) -> None:
        self._model = model
        self._controls = controls

    @property
    def version(self) -> str:
        return self._model.version

    def predict(self, text: str) -> Prediction:
        if self._controls.model_outage.is_set():
            raise RuntimeError("simulated model outage (demo control)")
        return self._model.predict(text)
