# Playlist & Queue Management

## Check how many videos are queued

To get a summary of the current video queue (pending count, played count, replay mode), use `get_queue_status`.

```
get_queue_status()
```

Returns:
- `pending_count`: videos waiting to be played
- `played_count`: videos already played
- `is_replaying`: `true` when the queue is empty and the stream is looping through played videos

The same data is available as a resource:

```
virtual-streamer://queue/status
```

---

## Browse the playlist

To list all entries in a programmation's playlist, use `get_playlist`.

```
get_playlist(programmation_id="prog-abc123")
```

To filter by status (`pending`, `playing`, `played`, `skipped`):

```
get_playlist(programmation_id="prog-abc123", status_filter="pending")
```

---

## Get the next video to play

To fetch the video that should play next, use `get_next_video`. It returns pending videos first, then falls back to a random already-played video.

```
get_next_video()
```

---

## Mark a video as played

After a video finishes, mark it so the queue advances correctly:

```
mark_video_played(entry_id="playlist-entry-id")
```

---

## Look up the active programmation

To get the programmation currently scheduled for this time slot, use `get_active_programmation`.

```
get_active_programmation()
```

To list all programmations for the stream:

```
list_programmations()
```

Stream and programmation details are also available as a resource:

```
virtual-streamer://stream/config
```
