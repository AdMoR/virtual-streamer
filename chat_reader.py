import twitch, json

cred_file = "/home/amor/twitch_creds_allo_jesus.json"
message_directory = "."
creds = json.load(open(cred_file))


def append_question(message: twitch.chat.Message):
    # !/usr/bin/python
    with open(f"{message_directory}/chat_log.txt", "a") as f:
        f.write(f"{message.user}|{message.text}\n")
    with open(f"{message_directory}/chat_log_saved.txt", "a") as f:
        f.write(f"{message.user}|{message.text}\n")


def handle_message(message: twitch.chat.Message) -> None:
    #if message.text.__contains__("!allo"):
    append_question(message)
    message.chat.send(f"Merci {message.user}, ta question est en cours de traitement.")
    print("Answer done")


def main_exec():
    chat = twitch.Chat(channel='allojesuschrist',
                       nickname='allojesuschrist',
                       oauth=creds["token"],
                       helix=twitch.Helix(client_id=creds["clientID"],
                                          client_secret=creds["clientSecret"]))

    chat.subscribe(handle_message)


if __name__ == "__main__":
    main_exec()