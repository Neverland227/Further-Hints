"""Permanently disabled G6K backend."""


class DeferredOutOfScope(RuntimeError):
    """Raised for mechanisms explicitly excluded by the experiment protocol."""


class G6KSieveSource:
    """Disabled interface stub; no import or execution of G6K occurs."""

    def prepare(self, instance, baseline_config):  # noqa: ANN001
        del instance, baseline_config

    def generate(self, budget, rng):  # noqa: ANN001
        del budget, rng
        raise DeferredOutOfScope("DEFERRED_OUT_OF_SCOPE: G6K execution is prohibited by protocol")

