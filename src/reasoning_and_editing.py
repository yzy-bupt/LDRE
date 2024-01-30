import json
import time
import traceback

from tqdm import tqdm
import openai
from pathlib import Path

openai.api_key = "your_api_key"

DATASET = 'circo' # 'cirr', 'fashioniq'

if DATASET == 'circo':
    SPLIT = 'test'
    input_json = 'CIRCO/annotations/test.json'
    dataset_path = Path('CIRCO')
elif DATASET == 'cirr':
    SPLIT = 'test1'
    input_json = 'CIRR/cirr/captions/cap.rc2.test1.json'
    dataset_path = Path('CIRR')

BLIP2_MODEL = 'opt' # or 'opt' or 't5'
MULTI_CAPTION = True
NUM_CAPTION = 15
with open(input_json, "r") as f:
    annotations = json.load(f)

for ans in tqdm(annotations):
    if DATASET == 'circo':
        rel_cap = ans["relative_caption"]
    elif DATASET == 'cirr':
        rel_cap = ans["caption"]
    if MULTI_CAPTION:
        blip2_caption = ans["multi_caption_{}".format(BLIP2_MODEL)]
    else:
        if BLIP2_MODEL == 'none':
            blip2_caption = ans["shared_concept"]
        elif BLIP2_MODEL == 'opt':
            blip2_caption = ans["blip2_caption"]
        else:
            blip2_caption = ans["blip2_caption_{}".format(BLIP2_MODEL)]

    sys_prompt = "I have an image. Given an instruction to edit the image, carefully generate a description of the edited image."

    if MULTI_CAPTION:
        multi_gpt = []
        for cap in blip2_caption:
            usr_prompt = "I will put my image content beginning with \"Image Content:\". The instruction I provide will begin with \"Instruction:\". The edited description you generate should begin with \"Edited Description:\". You just generate one edited description only begin with \"Edited Description:\". The edited description needs to be as simple as possible and only reflects image content. Just one line.\nA example:\nImage Content: a man adjusting a woman's tie.\nInstruction: has the woman and the man with the roles switched.\nEdited Description: a woman adjusting a man's tie.\n\nImage Content: {}\nInstruction: {}\nEdited Description:".format(cap, rel_cap)
            # print(prompt)
            while True:
                try:
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system",
                                    "content": sys_prompt},
                                    {"role": "user", "content": usr_prompt}],
                        timeout=10, request_timeout=10)
                    ret = completion['choices'][0]["message"]["content"].strip('\n')
                    multi_gpt.append(ret)
                    break
                except:
                    traceback.print_exc()
                    time.sleep(3)
        ans["multi_gpt-3.5_{}".format(BLIP2_MODEL)] = multi_gpt
        
    else:
        usr_prompt = "I will put my image content beginning with \"Image Content:\". The instruction I provide will begin with \"Instruction:\". The edited description you generate should begin with \"Edited Description:\". You just generate one edited description only begin with \"Edited Description:\". The edited description needs to be as simple as possible and only reflects image content. Just one line.\nA example:\nImage Content: a man adjusting a woman's tie.\nInstruction: has the woman and the man with the roles switched.\nEdited Description: a woman adjusting a man's tie.\n\nImage Content: {}\nInstruction: {}\nEdited Description:".format(blip2_caption, rel_cap)
        # print(prompt)
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system",
                                "content": sys_prompt},
                                {"role": "user", "content": usr_prompt}],
                    timeout=10, request_timeout=10)
                ret = completion['choices'][0]["message"]["content"].strip('\n')
                if BLIP2_MODEL == 'opt':
                    ans["gpt-3.5-turbo"] = ret
                else:
                    ans["gpt-3.5-turbo_{}".format(BLIP2_MODEL)] = ret
                break
            except:
                traceback.print_exc()
                time.sleep(3)

    # with open("CIRCO/annotations/gpt3.5-temp.json", "a") as f:
    #     f.write(json.dumps(ans, indent=4) + '\n')

with open(input_json, "w") as f:
    f.write(json.dumps(annotations, indent=4))