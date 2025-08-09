import json


folder_location = 'D:\soccernetCaptionDataset'

#TODO: get all folders files

with open("Labels-caption.json", mode ="r", encoding="utf-8") as read_file:
    json_file = json.load(read_file)

goal_captions = []

for annotation in json_file['annotations']:
    if annotation["label"] == "soccer-ball":
        goal_captions.append(annotation)
       #goal_captions.append(annotation["identified"])

print(*goal_captions, sep='\n')


