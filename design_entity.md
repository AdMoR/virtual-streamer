# Design: Character Creation with Video Upload

This document outlines the steps to extend the existing Entity Management Service so that new Characters can include an uploaded representative video clip.

## 1. Extend the Pydantic Model
- Update `CharacterCreate` model to include an optional `video_file: UploadFile` field.
- Ensure FastAPI knows to parse multipart/form-data for this endpoint.

## 2. Modify the Endpoint Signature
- Change the `@app.post("/characters")` handler to accept form data and file:
  ```python
  async def create_character(
      name: str = Form(...),
      description: str = Form(...),
      voice_samples: List[str] = Form(...),
      video_file: UploadFile = File(None)
  ):
  ```
- Remove the direct dependency on `CharacterCreate` for body parsing.

## 3. Store the Uploaded Video
- Generate a storage key, e.g. `characters/videos/{character_id}.{extension}`.
- Use or extend `AsyncS3Client` / `LocalFSClient` to support binary file uploads (e.g., `s3_put_file`).
- Write the raw bytes of `video_file` to the chosen backend.

## 4. Update the Character Model
- Add a new field `video_clip_path: Optional[str]` to `Character` and `CharacterBase`.
- After upload, populate `video_clip_path` with the S3/local URL or key.

## 5. Persist and Return Full Record
- Store the updated character record (including `video_clip_path`) via `s3_put_json`.
- Return the enriched `Character` instance in the response.

## 6. Client and Front-End Changes
- Ensure any UI or client code calls the new multipart endpoint.
- Provide a file input alongside existing character metadata fields.

## 7. Testing and Validation
- Write integration tests to:
  - Upload a small video file.
  - Retrieve the character and verify `video_clip_path` is set.
  - Confirm file exists in the backend store.

## 8. Cleanup and Error Handling
- Validate file type/size before storage.
- Delete uploaded file on any downstream failure before persisting metadata.
