# RunPod Worker (Queue-Based)

This worker is used by RunPod Serverless queue-based endpoints.

## Input

```json
{
  "input": {
    "story": "A short story text",
    "tone": "joyful",
    "speaker": "Lea"
  }
}
```

## Output

```json
{
  "audio_base64": "...",
  "duration": 12.34,
  "speaker": "Lea",
  "tone": "joyful"
}
```

## Supported Speakers

- Jon
- Lea
- Gary
- Jenna
