from pprint import pprint
import os.path as osp
import json 
import asl 
import os 


hand_ext = asl.HandMarksExtractor()

global_path = "./WLASL100/train"


def from_sample_of_id(sample_path: str, id: str = "00618"):
    data = {"id": id, "features": []}

    path = osp.join(sample_path, id)
    imgs = sorted(os.listdir(path), key=lambda img: int(img.split("_")[1].split(".")[0]))
    
    for img in imgs:
        try:
            hands = hand_ext.from_image(osp.join(path, img))
            data["features"].append(hands.to_numpy().tolist())
        except asl.OneORMoreThanTwoHandsError: pass 

    return data


def get_data_from_word(word: str, label: int = 0):
    data = {"word": word, "label": label, "samples": []}

    path = osp.join(global_path, word)

    for sample_id in os.listdir(path):
        ds = from_sample_of_id(path, sample_id)
        data["samples"].append(ds)

    return data

def get_data_for_all_words(root, split_set: str = "train"):

    #data = {split_set: []}

    path = osp.join(root, split_set)

    for label, word in enumerate(os.listdir(path)):

        dw = get_data_from_word(word, label)

        json.dump(dw, open(f"./dataset/{split_set}/{word}.json", "w"))

        print(f"{label} -> {word} done ", "."*10)


        #json.dump(data, open(f"./{split_set}.json", "w"))        


def write_json(root="./WLASL100/", split_set: str = "train"):
    
    path = osp.join(root, split_set)

    data = get_data_for_all_words(path)

    json.dump(data, open(f"./{split_set}.json", "w"))




if __name__ == "__main__":

    get_data_for_all_words("./WLASL100/", split_set="train")




