from py_crane.enums import Change


def test_types():
    assert Change.POS == Change(1)
    assert Change["POS"] in Change.ALL
    assert Change.ROT == Change(2)
    assert Change["ROT"] in Change.ALL
    ch = 0
    ch = Change.POS.value
    assert Change.ROT not in Change(ch)
    assert Change.POS in Change(ch)
    ch += Change.ROT.value
    assert Change.ROT in Change(ch)


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    test_types()
