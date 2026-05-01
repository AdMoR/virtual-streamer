# Content Inspiration

## Get news articles to inspire video topics

To fetch recent French news articles that can serve as inspiration for video content, use `fetch_news`.

```
fetch_news()
```

Returns a list of articles with:
- `title`: article headline
- `summary`: short summary
- `source`: news outlet name
- `link`: URL to the full article

Use these titles directly with `create_video_from_broadcast` or `create_video` to generate topical content during the stream.
