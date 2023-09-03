import twitch, json
import requests

cred_file = "/home/amor/twitch_creds_allo_jesus.json"

creds = json.load(open(cred_file))

# Replace these with your actual credentials
client_id = creds["clientID"]
client_secret = creds["clientSecret"]
redirect_uri = "http://localhost:8501/redirect"  # This should match the one you've set up in your Twitch application settings

# Step 1: Get an authorization code
authorization_url = f"https://id.twitch.tv/oauth2/authorize?client_id={client_id}&redirect_uri={redirect_uri}&response_type=code&scope=chat:read+chat:edit"
print("Visit the following URL to authorize the application:")
print(authorization_url)
authorization_code = input("Enter the authorization code from the URL: ")

# Step 2: Exchange authorization code for access token
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

if "access_token" in response_data:
    access_token = response_data["access_token"]
    print("Access Token:", access_token)
else:
    print("Error retrieving access token:", response_data)

creds["token"] = access_token

json.dump(creds, open(cred_file, "w"))
