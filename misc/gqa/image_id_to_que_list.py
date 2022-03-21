import json
from tqdm import tqdm
file_path = "/mnt/data1/guoyuyu/datasets/gqa/questions/"

def ImageId2QuestionId(split):
    image2que = {}
    ques_data = json.load(open(file_path+split+"_balanced_questions.json","r"))
    for i in tqdm(ques_data):
        instance = ques_data[i]
        imageId = instance["imageId"]
        if imageId not in image2que:
            image2que[imageId] = []
        image2que[imageId].append(i)
    with open(file_path+split+"_imageid2questionid.json", 'w') as outfile:
        json.dump(image2que, outfile)
    return
def check_map(split):
    ttl = 0
    ques_data = json.load(open(file_path + split + "_balanced_questions.json", "r"))
    imageids = json.load(open(file_path+split+"_images.json","r"))
    image2ques = json.load(open(file_path+split+"_imageid2questionid.json","r"))
    for i in imageids:
        ttl += len(image2ques[i])
    print(len(imageids), ttl, len(ques_data))


if __name__ == "__main__":
    # ImageId2QuestionId("train")
    # ImageId2QuestionId("val")
    check_map("val")
    check_map("train")


