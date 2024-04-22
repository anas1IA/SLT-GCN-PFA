import os.path as osp
import json
import asl
import os


SAVE_PATH = "./dataset/raw/"

if not osp.exists(SAVE_PATH): os.mkdir(SAVE_PATH)

BASE_PATH = "./ASL/train/"


def extract_featrs_from_one_sample(base_path, char: str = "A", label: int = 0):

    data = {"char": char, "label": label, "samples": []}

    char_path = osp.join(base_path, char)

    for sample in os.listdir(char_path)[:1000]:

        hand = asl.Hand()

        try:
            
            tmp = {"id": sample.split(".")[0], "x": []}

            path = osp.join(char_path, sample)

            hand.from_image(path)

            tmp["x"] = hand.x.tolist()

            data["samples"].append(tmp)

        except: pass 
        

    json.dump(data, open(f"{osp.join(SAVE_PATH, char)}.json", "w"))

    print(f"{label} -> done with char: {char}", "."*10)


def extract_for_all_char():

    chars = os.listdir(BASE_PATH)

    for label, char in enumerate(chars):

        extract_featrs_from_one_sample(BASE_PATH, char, label)





if __name__ == "__main__":

    extract_for_all_char()  

