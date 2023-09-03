import twitch, json
from utils import add_to_queue

cred_file = "/home/amor/twitch_creds_allo_jesus.json"
message_directory = "."
creds = json.load(open(cred_file))


def append_question(message: twitch.chat.Message):
    # !/usr/bin/python
    for queue_name in ["chat_log", "chat_log_saved"]:
        add_to_queue(queue_name, f"{message.user}|{message.text}")


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