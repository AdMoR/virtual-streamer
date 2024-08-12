import twitch, json
from utils import add_to_queue, ChatQuestion
from prompts import VERY_SARCASTIC_STANDUP_PROMPT


cred_file = "/home/amor/twitch_creds_allo_jesus.json"
message_directory = "."
creds = json.load(open(cred_file))


def append_question(message: twitch.chat.Message):
    # !/usr/bin/python
    queue_name = "chat_log"
    # API : name: str,  question: str, routing_queue: str = "video_response_queue", prompt: str = None
    msg = ChatQuestion(name=message.sender, question=message.text, routing_queue=queue_name,
                       prompt=VERY_SARCASTIC_STANDUP_PROMPT, next_queue="obs")
    add_to_queue(queue_name, msg.serialize())


def handle_message(message: twitch.chat.Message) -> None:
    if message.text.lstrip(" ").lower().startswith("!allo") or message.text.lstrip(" ").lower().startswith("allo"):
        append_question(message)
        message.chat.send(f"Merci {message.user}, ta question est en cours de traitement.")
        print("Answer done")
    elif message.text.lower().startswith("jesus") or message.text.lower().startswith("jésus"):
        message.chat.send(f"{message.user}, si tu souhaites poser une question à Jésus. "
                          f"Utilises !allo avant ta question. Ex: !allo Salut, la forme ?.")


def main_exec():
    chat = twitch.Chat(channel='allojesuschrist',
                       nickname='allojesuschrist',
                       oauth=creds["token"],
                       helix=twitch.Helix(client_id=creds["clientID"],
                                          client_secret=creds["clientSecret"]))

    chat.subscribe(handle_message)


if __name__ == "__main__":
    main_exec()