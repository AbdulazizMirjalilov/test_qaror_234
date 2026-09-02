from bs4 import BeautifulSoup

from app.ingestion.loader import table_to_markdown


def make_table(html: str):
    return BeautifulSoup(html, "lxml").find("table")


def test_simple_table_to_markdown():
    table = make_table("<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>")
    md = table_to_markdown(table)
    lines = md.split("\n")
    assert lines[0] == "| A | B |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| 1 | 2 |"


def test_ragged_rows_padded():
    table = make_table("<table><tr><td>A</td><td>B</td><td>C</td></tr><tr><td>1</td></tr></table>")
    md = table_to_markdown(table)
    assert md.split("\n")[2] == "| 1 |  |  |"


def test_pipe_characters_escaped():
    table = make_table("<table><tr><td>a|b</td></tr><tr><td>x</td></tr></table>")
    assert "a\\|b" in table_to_markdown(table)


def test_empty_table_returns_empty_string():
    assert table_to_markdown(make_table("<table></table>")) == ""
