from search.parse_osbooks import parse_cnxml_file


def test_intro_module_parses():
    chunks = parse_cnxml_file("tests/fixtures/m53472.cnxml", "calculus-volume-1")
    assert chunks, "expected chunks from m53472"
    assert all(c.module_id == "m53472" for c in chunks)
    assert all(c.source == "osbooks" for c in chunks)
    assert chunks[0].title == "Introduction"
    assert all(c.text and "<" not in c.text for c in chunks), "XML tags must be stripped"
    assert any("Calculus is the mathematics" in c.text for c in chunks)


def test_math_module_extracts_equations():
    chunks = parse_cnxml_file("tests/fixtures/m53481.cnxml", "calculus-volume-1")
    assert chunks, "expected chunks from m53481"
    assert chunks[0].title == "Exponential and Logarithmic Functions"
    assert any(c.equations for c in chunks), "expected MathML equations linearized"
    assert any("Exponential Functions" in c.section for c in chunks), "section titles preserved"

def test_worked_example_keeps_solution():
    chunks = parse_cnxml_file("tests/fixtures/m53481.cnxml", "calculus-volume-1")
    assert any("2828" in c.text for c in chunks), "bacteria example solution was dropped"

def test_exponents_keep_structure():
    chunks = parse_cnxml_file("tests/fixtures/m53481.cnxml", "calculus-volume-1")
    assert any("2^(t/4)" in c.text for c in chunks), "exponent structure was lost"
    assert any("2^(6/4)" in c.text and "2828" in c.text for c in chunks)
    assert not any("1000·2t/4" in c.text for c in chunks), "flattened exponent is back"