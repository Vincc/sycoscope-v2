from judging.judge_reddit_edits import parse


def test_parse_reads_last_line_and_rejects_reaction_without_edit():
    assert parse("The post has an EDIT: section.\n\nEDIT=1 REACTION=0") == (1, 0)
    assert parse("EDIT=1 REACTION=1\nmore text") == (None, None)
    assert parse("EDIT=0 REACTION=1") == (None, None)
    assert parse(None) == (None, None)
