"""Tests for the answer prompt."""

from doc_qa.answer.prompt import NO_TEXT, SYSTEM_PROMPT, PageInput, build_messages, retry_messages


def test_pages_are_labelled_in_order_with_image_and_text():
    pages = [PageInput("d/7", "img7", "first page"), PageInput("d/2", "img2", "  ")]
    system, user = build_messages("What is X?", pages, max_text_chars=5)
    assert system["content"][0]["text"] == SYSTEM_PROMPT
    content = user["content"]
    assert [item["type"] for item in content] == ["text", "image", "text", "text", "image", "text", "text"]
    assert content[0]["text"] == "[Page 1]" and content[1]["image"] == "img7"
    assert content[2]["text"] == "Text of page 1:\nfirst"  # cut to 5 characters
    assert content[3]["text"] == "[Page 2]" and content[5]["text"] == f"Text of page 2:\n{NO_TEXT}"
    assert content[-1]["text"] == "Question: What is X?"


def test_retry_shows_the_rejected_reply_and_the_reason():
    messages = build_messages("Q?", [PageInput("d/1", "img", "t")], 100)
    retried = retry_messages(messages, "not json", "it contains no JSON object")
    assert retried[:2] == messages
    assert retried[2] == {"role": "assistant", "content": [{"type": "text", "text": "not json"}]}
    assert "it contains no JSON object" in retried[3]["content"][0]["text"]
    assert len(messages) == 2  # the original chat is unchanged
