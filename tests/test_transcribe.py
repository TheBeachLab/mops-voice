from mops_voice.transcribe import is_gibberish, HALLUCINATION_BLOCKLIST


def test_is_gibberish_empty():
    assert is_gibberish("") is True


def test_is_gibberish_short():
    assert is_gibberish("hi") is True  # < 3 chars


def test_is_gibberish_valid():
    assert is_gibberish("mill the PCB traces") is False


def test_is_gibberish_hallucination():
    assert is_gibberish("Thank you for watching") is True


def test_hallucination_blocklist_is_lowercase():
    for entry in HALLUCINATION_BLOCKLIST:
        assert entry == entry.lower()
