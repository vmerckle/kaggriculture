# Replay inputs

This repository does not distribute a replay pack, episode index, packed corpus, or trained checkpoints.
Readers must independently obtain permitted Kaggriculture game records and prepare the local format described here.
There is no downloader or raw Kaggle replay converter in this repository.
Follow Kaggle's applicable access and redistribution terms when obtaining records.

## Local layout

Pass a directory containing index.json and gzipped compact replays to corpus.py.
Each episode's path is relative to that directory, for example replays/episode-ID-compact.json.gz.

The index is a JSON object with an episodes array.
Each episode entry needs:

- episode_id: integer game identifier.
- create_time: ISO timestamp, used for recent-game weighting and submission ordering.
- state: COMPLETED for games eligible for selection.
- on_pin: boolean indicating that the record belongs to the pinned engine version.
- path: relative path to the gzipped compact replay.
- agents: two entries with index (seat 0 or 1), reward (final money; episodes with a null reward are skipped), submission_id, and team_id.

Two optional top-level fields of the index choose what is trained; the code has no team list of its own:

- teams: team IDs to clone; without it, corpus.py clones every team in the eligible games, so an index holding only the wanted teams' games needs no list.
- finetune_team: the team ID to fine-tune on and export as the PPO starting player; without it, the selected team with the most won training seats.

Labels are t followed by the team ID; OUT/corpus/train/meta.json records the team order and the chosen finetune_team.

Each compact replay is a gzipped JSON object containing seed and actions.
actions is a chronological array of two-seat action pairs.
Its first entry is the initial non-decision entry; corpus.py drops it before replaying the decision hours.
The remaining entries must contain both seats' engine action dictionaries for each decision hour.
This is the corpus builder's input format, not a claim that downloaded Kaggle records already have this shape.

Do not set on_pin merely to force a record into the corpus.
Verify engine provenance before preparing the index.
The builder also requires the recorded actions and their encoded/decoded equivalents to reproduce both recorded final rewards exactly; failing seats are recorded as errors and skipped.

## Building your corpus

    python corpus.py /path/to/local/replay-input out/corpus

The default worker count is the detected CPU count minus one.
Use --workers N to limit CPU and memory use.
Allow about 5 MB of SSD space per selected seat.
