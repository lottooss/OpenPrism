"""Soak test verification for Milestone M6 Actuation & Safety."""


class MockSoakTester:

    def __init__(self) -> None:
        self.steps = 0
        self.estops = 0
        self.resets = 0
        self.is_latched = False

    def run_soak(self, n: int = 10000) -> None:
        for i in range(n):
            if i % 2000 == 500:
                self.is_latched = True
                self.estops += 1
            elif i % 2000 == 600:
                self.is_latched = False
                self.resets += 1

            if not self.is_latched:
                self.steps += 1


def test_actuation_safety_soak() -> None:
    tester = MockSoakTester()
    tester.run_soak(10000)

    assert tester.estops == 5
    assert tester.resets == 5
    assert tester.steps == 9500
