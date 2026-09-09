from datetime import UTC, datetime

from src.data import Tweet, ancestor_path, build_episodes, parse_tweet, scrub_text, time_split


def tweet(identifier, author, inbound, hour, text, *, parent=None, replies=()):
    return Tweet(
        tweet_id=str(identifier),
        author_id=author,
        inbound=inbound,
        created_at=datetime(2017, 1, 1, hour, tzinfo=UTC),
        text=text,
        response_ids=tuple(str(item) for item in replies),
        in_response_to=str(parent) if parent else None,
    )


def test_scrub_preserves_brand_and_small_version_numbers():
    scrubbed = scrub_text("DM @Other and @BrandHelp at a@b.com, call 9876543210; v2.4", "BrandHelp")
    assert scrubbed == "DM <USER> and @brandhelp at <EMAIL>, call <PHONE>; v2.4"


def test_parse_tweet_accepts_dataset_timestamp_format():
    parsed = parse_tweet(
        {
            "tweet_id": "1",
            "author_id": "brand",
            "inbound": "False",
            "created_at": "Tue Oct 31 22:10:47 +0000 2017",
            "text": "hello",
            "response_tweet_id": "2,3",
            "in_response_to_tweet_id": "4",
        }
    )
    assert parsed.created_at.year == 2017
    assert parsed.response_ids == ("2", "3")


def test_ancestor_path_detects_future_parent_and_cycle():
    future = {"1": tweet(1, "brand", False, 2, "late"), "2": tweet(2, "u", True, 1, "now", parent=1)}
    assert ancestor_path(future["2"], future) == ([], "future_parent")
    cycle = {"1": tweet(1, "brand", False, 1, "a", parent=2), "2": tweet(2, "u", True, 2, "b", parent=1)}
    assert ancestor_path(cycle["2"], cycle) == ([], "cycle")


def test_builds_one_episode_per_component_and_never_uses_future_reply_as_context():
    tweets = {
        "1": tweet(1, "brand", False, 1, "How can we help?", replies=(2,)),
        "2": tweet(2, "u", True, 2, "My app crashes", parent=1, replies=(3, 4)),
        "3": tweet(3, "brand", False, 3, "Try updating", parent=2),
        "4": tweet(4, "u", True, 4, "Still broken", parent=2),
    }
    episodes, excluded = build_episodes(tweets, "brand")
    assert len(episodes) == 1
    assert episodes[0].message_id == "2"
    assert episodes[0].historical_reply_id == "3"
    assert [turn["message_id"] for turn in episodes[0].context] == ["1"]
    assert excluded["additional_message_in_component"] == 1


def test_time_split_is_chronological_and_complete():
    # Construct directly through episodes is not needed; assert split boundaries with real generated records.
    from src.data import Episode
    records = [Episode(str(i), str(i), f"2017-01-{i:02d}T00:00:00+00:00", "brand", "x", (), None, None) for i in range(10)]
    split = time_split(records)
    assert [item.message_id for item in split["train"]] == [str(i) for i in range(6)]
    assert [item.message_id for item in split["dev"]] == ["6", "7"]
    assert [item.message_id for item in split["test_pool"]] == ["8", "9"]
