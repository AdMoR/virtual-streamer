# Viewer Interaction

## Greet a new viewer

To generate a personalized greeting video when a viewer joins the channel for the first time, use `greet_viewer`.

```
greet_viewer(user_name="CoolViewer42")
```

The agent analyzes the username and generates a TTS + lip-sync video. The default character is `jesus_short`. To use a different character:

```
greet_viewer(user_name="CoolViewer42", character_id="narrator")
```

The video is automatically added to the stream playlist.

---

## Answer a viewer's question

To generate a video responding to a question from chat, use `answer_viewer_question`.

```
answer_viewer_question(
    question="What is the meaning of life?",
    user_name="CoolViewer42"
)
```

The agent generates a sarcastic answer and creates a TTS + lip-sync video added to the playlist. You can specify a different character:

```
answer_viewer_question(
    question="Is pizza a vegetable?",
    user_name="PizzaFan99",
    character_id="jesus_short"
)
```

---

## Check content safety before acting on a viewer message

To verify that user-submitted text is safe before generating a response, use `check_content_safety`.

```
check_content_safety(text="Can you explain quantum entanglement?")
```

Returns a classification of `NORMAL` or `MALICIOUS` with a justification. Always run this before processing viewer-submitted titles or questions.
