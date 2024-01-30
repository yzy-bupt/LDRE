import json
from pathlib import Path
import torch
from tqdm import tqdm
import PIL.Image as Image
from lavis.models import load_model_and_preprocess

DATASET = 'circo' # 'cirr', 'fashioniq'

if DATASET == 'circo':
    dataset_path = Path('CIRCO')
    SPLIT = 'test'
    with open(dataset_path / 'annotations' / f'{SPLIT}.json', "r") as f:
        annotations = json.load(f)

    with open(dataset_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
        imgs_info = json.load(f)

    img_paths = [dataset_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                        imgs_info["images"]]
    img_ids = [img_info["id"] for img_info in imgs_info["images"]]
    img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(img_ids)}

elif DATASET == 'cirr':
    dataset_path = Path('CIRR')
    SPLIT = 'test1'
    with open(dataset_path  / 'cirr' / 'captions' / f'cap.rc2.{SPLIT}.json') as f:
        annotations = json.load(f)

    with open(dataset_path / 'cirr' / 'image_splits' / f'split.rc2.{SPLIT}.json') as f:
        name_to_relpath = json.load(f)
else:
    dataset_path = Path('FashionIQ')
    SPLIT = 'val'
    DRESS = 'toptee' # 'shirt', 'toptee'
    new_annotations = []
    with open(dataset_path / 'captions' / f'new_cap.{DRESS}.{SPLIT}.json') as f:
        annotations = json.load(f)

BLIP2_MODEL = 'opt' # or 'opt'
MULTI_CAPTION = True
NUM_CAPTION = 15

# output_json = '{}_{}_multi.json'.format(SPLIT, BLIP2_MODEL) if MULTI_CAPTION else '{}_{}.json'.format(SPLIT, BLIP2_MODEL)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

if BLIP2_MODEL == 'opt':
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
    )
else:
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
    )
# model = model.float()

for ans in tqdm(annotations):
    if DATASET == 'circo':
        ref_img_id = ans["reference_img_id"]
        reference_img_id = str(ref_img_id)
        reference_img_path = img_paths[img_ids_indexes_map[reference_img_id]]
    elif DATASET == 'cirr':
        ref_img_id = ans["reference"]
        rel_cap = ans["caption"]
        reference_img_path = dataset_path / name_to_relpath[ref_img_id]
    else:
        ref_img_name = ans["candidate"] + '.jpg'
        reference_img_path = dataset_path / 'images' / ref_img_name

    raw_image = Image.open(reference_img_path).convert('RGB')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    if MULTI_CAPTION:
        caption = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=NUM_CAPTION)
    else:
        caption = model.generate({"image": image})
    # print(caption)

    if MULTI_CAPTION:
        ans["multi_caption_{}".format(BLIP2_MODEL)] = caption
    else:
        ans["blip2_caption_{}".format(BLIP2_MODEL)] = caption[0]

    if DATASET == 'fashioniq':
        new_annotations.append(ans)
    # with open("CIRCO/annotations/blip2_caption_t5.json", "a") as f:
    #     f.write(json.dumps(ans, indent=4) + '\n')

if DATASET == 'circo':
    with open("CIRCO/annotations/{}".format(f'{SPLIT}.json'), "w") as f:
        f.write(json.dumps(annotations, indent=4))
elif DATASET == 'cirr':
    with open("CIRR/cirr/captions/{}".format(f'cap.rc2.{SPLIT}.json'), "w") as f:
        f.write(json.dumps(annotations, indent=4))
else:
    with open("FashionIQ/captions/" + f'cap.{DRESS}.{SPLIT}.json', "w") as f:
        f.write(json.dumps(new_annotations, indent=4))
