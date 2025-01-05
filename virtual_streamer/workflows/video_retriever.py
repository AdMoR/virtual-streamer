import os
import json
from virtual_streamer.utils.utils import get_length
from llama_index.core.schema import TextNode


def load_json_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'rb') as file:
                data = json.load(file)
                documents.append(data)
    return documents


# Index the JSON documents into the LlamaIndex Rag DB
def prepare_nodes(json_documents):
    """
    to check = https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_metadata_extractor/
    """
    nodes = list()
    for i, doc in enumerate(json_documents):
        base_video_name, scene_index, *args = os.path.basename(doc["path"]).split('-Scene-')
        doc["description_full"] = " ".join(list(x.values())[0] for x in doc["description"])
        doc["who"] = doc["who"][0][0] if len(doc["who"]) >= 1 else ""
        #del better_doc["path"]
        document_str = list()
        for k in doc:
            if k in {"path", "description", "who", "duration"}:
                continue
            sub_str = f"<{k}>{doc[k]}<\\{k}>"
            document_str.append(sub_str)
        final_str = "\n".join(document_str)
        nodes.append(TextNode(text=final_str,
                              metadata={"who": doc["who"],
                                        "base_video": base_video_name,
                                        "scene_index": scene_index,
                                        "path": doc["path"],
                                        "duration": get_length(doc["path"])
                                        if "duration" not in doc else doc["duration"]},
                              id_=str(hash(doc["description_full"]))))

    return nodes
