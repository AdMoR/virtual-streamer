import json
import webbrowser
import requests

creds = json.load(open("./creds.json"))
client_id = creds["client_id"]
client_secret = creds["client_secret"]

redirect_uri = "http://localhost:8501/redirect"  # This should match the redirect URI set in your Twitch app settings
scopes = "user:bot user:read:chat user:write:chat chat:edit chat:read"  # Add the scopes you need

auth_url = (
    f"https://id.twitch.tv/oauth2/authorize"
    f"?client_id={client_id}"
    f"&redirect_uri={redirect_uri}"
    f"&response_type=code"
    f"&scope={scopes}"
)

print("Please go to this URL and authorize the application:")
print(auth_url)

# Open the URL in the default web browser
webbrowser.open(auth_url)

authorization_code = input("What is the code ?")

token_url = "https://id.twitch.tv/oauth2/token"
payload = {
    "client_id": client_id,
    "client_secret": client_secret,
    "code": authorization_code,
    "grant_type": "authorization_code",
    "redirect_uri": redirect_uri
}
response = requests.post(token_url, data=payload)
response_data = response.json()

print(response_data)
creds["refresh_token"] = response_data["refresh_token"]
json.dump(creds, open("creds.json", "w"))
