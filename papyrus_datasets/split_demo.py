import json

def get_json():
    with open("./random_split.json") as file:
        json_file = file.read()

    return json.loads(json_file)

def get_demo(json_obj):
    targets = ["P00918","P03372","P04637","P08684","P14416","P22303","P42336","Q12809","Q16637","Q9Y468"]
    for target in targets:
        for seed in range(21):
            seed = str(seed)
            sets = json_obj["random"][target][seed]
            sets["train"] = sets["train"][:5]
            sets["valid"] = sets["valid"][:5]
            sets["test"] = sets["test"][:5]
    
    with open("./random_split_demo.json",mode="w") as file:
        file.write(json.dumps(json_obj, indent=1))

if __name__ == "__main__":
    json_file = get_json()
    get_demo(json_file)
    