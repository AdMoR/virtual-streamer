## What Changed in LTX-2.3

LTX-2.3 has a redesigned text connector architecture that makes it significantly more responsive to prompt details. This means:
More faithful prompt adherence

‍Specific descriptions of facial expressions, timing, pauses, and emotional beats translate more reliably into the output. You can direct acting at a granular level — "he pauses, looks to the side, then continues speaking with a cracking voice" — and expect the model to follow.
Prompt length matters

‍Longer, more descriptive prompts consistently outperform short ones on 2.3. If you're generating longer videos (8–10 seconds), make sure your prompt is detailed enough to fill the duration. A short prompt for a long video often results in the model rushing through the described action.
Break dialogue into segments

‍When prompting for speaking characters, break long sentences into shorter phrases with acting directions between them. For example:

A middle-aged man with greying hair speaks in a sad, slow-paced voice, "I remember after you kids came along..." He pauses and looks to the side, then continues, "your mom..." His eyes widen momentarily. He finishes with a cracking voice, "said something to me I never quite understood." The camera slowly zooms into his face. The audio is crisp with faint room tone.

This gives the model explicit direction on pacing, emotion, and physical acting for each beat.
Audio descriptions have more impact

‍With improved audio quality in 2.3, it's worth spending more attention on audio prompts. Describe the acoustic environment, the character's voice qualities, and any ambient sounds you want.
Key Elements to Include

When writing a prompt, aim to include the following elements:
1. Establish the Shot

Use cinematography terms that match your intended genre. Include shot scale or category-specific characteristics to refine the visual style.
2. Set the Scene

Describe lighting conditions, color palette, surface textures, and atmosphere to establish mood and tone.
3. Describe the Action

Write the core action as a natural sequence, flowing clearly from beginning to end.
4. Define the Character(s)

Include age, hairstyle, clothing, and distinguishing features. Express emotion through physical cues, not abstract labels.
5. Identify Camera Movement(s)

Specify how and when the camera moves. Describing how subjects appear after the movement helps the model complete the motion accurately.
6. Describe the Audio

Clearly describe ambient sound, music, speech, or singing.

    Place spoken dialogue in quotation marks
    Specify language and accent if needed

For Best Results

    Write your prompt as a single flowing paragraph
    Use present tense verbs for action and movement
    Match the level of detail to the shot scale — close-ups need more detail than wide shots
    Describe camera movement relative to the subject

## Tips by Use Case
#### Text-to-Video
Start with a strong visual description. Include subject, action, environment, lighting, camera movement, and audio. The model generates everything from scratch, so detail is your primary lever.

#### Image-to-Video
Focus your prompt on the motion and action you want — the visual starting point is already defined by your input image. Describe what happens next: how the subject moves, how the camera follows, what sounds emerge. Avoid describing the static elements already visible in the image. Instead, describe the transition from stillness to motion.

#### Audio-to-Video
Your audio input anchors the temporal structure. Use the prompt to describe the visual interpretation of that audio — what scenes, subjects, and camera work should accompany the soundtrack.


In this repository, most generations will be audio+txt+image to video generations.