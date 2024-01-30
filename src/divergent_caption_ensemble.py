import json
import pickle
from args import args_define
from typing import List, Tuple, Dict

import clip
import open_clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CIRRDataset, CIRCODataset
from utils import extract_image_features, device, collate_fn, PROJECT_ROOT, targetpad_transform


@torch.no_grad()
def cirr_generate_test_submission_file(dataset_path: str, clip_model_name: str, preprocess: callable, submission_name: str) -> None:
    """
    Generate the test submission file for the CIRR dataset given the pseudo tokens
    """

    # Load the CLIP model
    if clip_model_name == 'ViT-g-14':
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-g-14', device=device, pretrained='laion2b_s34b_b88k')
    else:
        clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().eval()

    # Compute the index features
    classic_test_dataset = CIRRDataset(dataset_path, 'test1', 'classic', preprocess)
    # if args.xxx:
    #     index_features = torch.load('feature/{}/index_features_G14.pt'.format(args.dataset))
    #     index_names = np.load('feature/{}/index_names_G14.npy'.format(args.dataset))
    #     index_names = index_names.tolist()
    # else:
    
    index_features, index_names = extract_image_features(classic_test_dataset, clip_model)

    relative_test_dataset = CIRRDataset(dataset_path, 'test1', 'relative', preprocess)

    # Get the predictions dicts
    pairid_to_retrieved_images, pairid_to_group_retrieved_images = \
        cirr_generate_test_dicts(relative_test_dataset, clip_model, index_features, index_names,
                                args.nums_caption)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_retrieved_images)
    group_submission.update(pairid_to_group_retrieved_images)

    submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'cirr'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{submission_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"subset_{submission_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def cirr_generate_test_dicts(relative_test_dataset: CIRRDataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str], nums_caption) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Generate the test submission dicts for the CIRR dataset given the pseudo tokens
    """

    # Get the predicted features
    predicted_features, reference_names, pairs_id, group_members = \
        cirr_generate_test_predictions(clip_model, relative_test_dataset)

    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = index_features.to(device)
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_retrieved_images = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                                  zip(pairs_id, sorted_index_names)}
    pairid_to_group_retrieved_images = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                        zip(pairs_id, sorted_group_names)}

    return pairid_to_retrieved_images, pairid_to_group_retrieved_images


def cirr_generate_test_predictions(clip_model: CLIP, relative_test_dataset: CIRRDataset,
                                   ) -> \
        Tuple[torch.Tensor, List[str], List[str], List[List[str]]]:
    """
    Generate the test prediction features for the CIRR dataset given the pseudo tokens
    """

    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=16,
                                      pin_memory=False)

    predicted_features_list = []
    reference_names_list = []
    pair_id_list = []
    group_members_list = []
    if args.type == 'G':
        tokenizer = open_clip.get_tokenizer('ViT-g-14')
    else:
        tokenizer = clip.tokenize

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        pairs_id = batch['pair_id']
        relative_captions = batch['relative_caption']
        group_members = batch['group_members']
        multi_caption = batch['multi_{}'.format(args.caption_type)]
        multi_gpt_caption = batch['multi_gpt_{}'.format(args.caption_type)]

        group_members = np.array(group_members).T.tolist()

        # input_captions = [
        #     f"a photo of $ that {rel_caption}" for rel_caption in relative_captions]
        if args.is_gpt_caption:
            input_captions = multi_gpt_caption

        else:
            if args.is_rel_caption:
                input_captions = [f"a photo that {caption}" for caption in relative_captions]
            else:
                input_captions = multi_caption[0]
        
        if args.multi_caption:
            text_features_list = []
            for cap in input_captions:
                tokenized_input_captions = tokenizer(cap, context_length=77).to(device)
                text_features = clip_model.encode_text(tokenized_input_captions)
                text_features_list.append(text_features)
            text_features_list = torch.stack(text_features_list)
            text_features = torch.mean(text_features_list, dim=0)

        else:
            tokenized_input_captions = tokenizer(input_captions, context_length=77).to(device)
            text_features = clip_model.encode_text(tokenized_input_captions)

        predicted_features = F.normalize(text_features)

        predicted_features_list.append(predicted_features)
        reference_names_list.extend(reference_names)
        pair_id_list.extend(pairs_id)
        group_members_list.extend(group_members)

    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, reference_names_list, pair_id_list, group_members_list


@torch.no_grad()
def circo_generate_test_submission_file(dataset_path: str, clip_model_name: str, preprocess: callable,
                                        submission_name: str) -> None:
    """
    Generate the test submission file for the CIRCO dataset given the pseudo tokens
    """

    # Load the CLIP model
    if clip_model_name == 'ViT-g-14':
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-g-14', device=device, pretrained='laion2b_s34b_b88k')
    else:
        clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().eval().requires_grad_(False)

    # Compute the index features
    classic_test_dataset = CIRCODataset(dataset_path, 'test', 'classic', preprocess)
    index_features, index_names = extract_image_features(classic_test_dataset, clip_model)

    relative_test_dataset = CIRCODataset(dataset_path, 'test', 'relative', preprocess)


    # for idx in range(15):
    #     # Get the predictions dict
    #     queryid_to_retrieved_images = circo_generate_test_dict(relative_test_dataset, clip_model, index_features,
    #                                                         index_names, ref_names_list, pseudo_tokens, idx + 1)

    #     submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'circo'
    #     submissions_folder_path.mkdir(exist_ok=True, parents=True)

    #     with open(submissions_folder_path / f"{submission_name}_{idx + 1}.json", 'w+') as file:
    #         json.dump(queryid_to_retrieved_images, file, sort_keys=True)

    # Get the predictions dict
    queryid_to_retrieved_images = circo_generate_test_dict(relative_test_dataset, clip_model, index_features,
                                                           index_names, args.nums_caption)

    submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'circo'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{submission_name}.json", 'w+') as file:
        json.dump(queryid_to_retrieved_images, file, sort_keys=True)


def circo_generate_test_predictions(clip_model: CLIP, relative_test_dataset: CIRCODataset,
                                    use_momentum_strategy=False, debiased_id=-1) -> [torch.Tensor, List[List[str]]]:
    """
    Generate the test prediction features for the CIRCO dataset given the pseudo tokens
    """

    # Create the test dataloader
    # num_workers
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=16,
                                      pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    query_ids_list = []
    if args.type == 'G':
        tokenizer = open_clip.get_tokenizer('ViT-g-14')
    else:
        tokenizer = clip.tokenize

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        relative_captions = batch['relative_caption']
        query_ids = batch['query_id']
        blip2_caption = batch['blip2_caption_{}'.format(args.caption_type)]
        gpt_caption = batch['gpt_caption_{}'.format(args.caption_type)]
        multi_caption = batch['multi_{}'.format(args.caption_type)]
        multi_gpt_caption = batch['multi_gpt_{}'.format(args.caption_type)]

        if args.is_gpt_caption:
            if args.multi_caption:
                if use_momentum_strategy:
                    if debiased_id != -1:
                        input_captions = multi_caption[debiased_id]
                    else:
                        input_captions = multi_caption
                else:
                    if debiased_id != -1:
                        input_captions = multi_gpt_caption[debiased_id]
                    else:
                        input_captions = multi_gpt_caption
            else:
                if use_momentum_strategy:
                    input_captions = blip2_caption
                else:
                    input_captions = [f"{caption}" for caption in gpt_caption]
        else:
            if args.multi_caption and args.is_rel_caption:
                input_captions = multi_caption
                for i in range(len(input_captions)): 
                    input_captions[i] = [f"a photo of {input_captions[i][inx]} that {relative_captions[inx]}" for inx in range(len(input_captions[i]))]
            else:
                input_captions = [f"a photo that {caption}" for caption in relative_captions]

        if args.multi_caption and debiased_id == -1:
            text_features_list = []
            for cap in input_captions:
                tokenized_input_captions = tokenizer(cap, context_length=77).to(device)
                text_features = clip_model.encode_text(tokenized_input_captions)
                text_features_list.append(text_features)
            text_features_list = torch.stack(text_features_list)
            text_features = torch.mean(text_features_list, dim=0)

        else:
            tokenized_input_captions = tokenizer(input_captions, context_length=77).to(device)
            text_features = clip_model.encode_text(tokenized_input_captions)
        predicted_features = F.normalize(text_features)

        predicted_features_list.append(predicted_features)
        query_ids_list.extend(query_ids)

    predicted_features = torch.vstack(predicted_features_list)
    return predicted_features, query_ids_list


def circo_generate_test_dict(relative_test_dataset: CIRCODataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str], nums_caption) \
        -> Dict[str, List[str]]:
    """
    Generate the test submission dicts for the CIRCO dataset given the pseudo tokens
    """

    # Get the predicted features
    if args.is_gpt_predicted_features:
        if args.use_debiased_sample:
            predicted_features_list = []
            for i in range(nums_caption):
                predicted_features = torch.load('feature/debiased/gpt_predicted_features_{}.pt'.format(i))
                predicted_features_list.append(predicted_features)
            query_ids = np.load('feature/query_ids.npy')

        else:
            predicted_features = torch.load('feature/gpt_predicted_features.pt')
            query_ids = np.load('feature/query_ids.npy')
    else:
        if args.use_debiased_sample:
            predicted_features_list = []
            for i in range(nums_caption):
                predicted_features, query_ids = circo_generate_test_predictions(clip_model, relative_test_dataset, debiased_id=i)
                torch.save(predicted_features, 'feature/debiased/gpt_predicted_features_{}.pt'.format(i))
                predicted_features_list.append(predicted_features)
        else:
            predicted_features, query_ids = circo_generate_test_predictions(clip_model, relative_test_dataset)
            if args.features_save_path:
                np.save('feature/query_ids.npy', query_ids)
                torch.save(predicted_features, 'feature/gpt_predicted_features.pt')
    
    # Normalize the features
    index_features = index_features.float().to(device)
    index_features = F.normalize(index_features, dim=-1)

    if args.use_momentum_strategy:
        if args.is_blip_predicted_features:
            if args.use_debiased_sample:
                blip_predicted_features_list = []
                for i in range(nums_caption):
                    blip_predicted_features = torch.load('feature/debiased/blip_predicted_features_{}.pt'.format(i))
                    blip_predicted_features_list.append(blip_predicted_features)
            else:
                blip_predicted_features = torch.load('feature/blip_predicted_features.pt')
        else:
            if args.use_debiased_sample:
                blip_predicted_features_list = []
                for i in range(nums_caption):
                    blip_predicted_features, _ = circo_generate_test_predictions(clip_model, relative_test_dataset, True, debiased_id=i)
                    torch.save(blip_predicted_features, 'feature/debiased/blip_predicted_features_{}.pt'.format(i))
                    blip_predicted_features_list.append(blip_predicted_features)
            else:
                blip_predicted_features, _ = circo_generate_test_predictions(clip_model, relative_test_dataset, True)
                if args.features_save_path:
                    torch.save(blip_predicted_features, 'feature/blip_predicted_features.pt')
        
        if args.use_debiased_sample:
            neg_diff_val = []
            for i in range(nums_caption):
                gpt_features = predicted_features_list[i]
                blip_features = blip_predicted_features_list[i]
                # neg_diff_val.append(torch.sum(1 - (gpt_features * blip_features)).item())

                similarity_after = gpt_features @ index_features.T
                similarity_before = blip_features @ index_features.T
                diff = similarity_after - similarity_before
                diff[diff > 0] = 0
                diff = -diff
                diff = torch.topk(diff, dim=-1, k=50).values
                sum_diff = torch.sum(diff)
                # sum_diff = torch.sum(diff < 0)
                neg_diff_val.append(sum_diff.item())

            neg_diff_val_tensor = torch.tensor(neg_diff_val).float().to(device)
            print(neg_diff_val_tensor)
            debiased_weight = torch.softmax(neg_diff_val_tensor / torch.max(neg_diff_val_tensor) / args.debiased_temperature, 0)
            print(debiased_weight)
            predicted_features_tensor = torch.stack(predicted_features_list)
            if 0:
                debiased_features = torch.mean(predicted_features_tensor, dim=0)
            else:
                debiased_features = torch.sum(predicted_features_tensor * debiased_weight.unsqueeze(1).unsqueeze(2), dim=0)
            similarity = debiased_features @ index_features.T


        else:
            similarity_after = predicted_features @ index_features.T
            similarity_before = blip_predicted_features @ index_features.T

            diff = similarity_after - similarity_before

            similarity = similarity_after + args.momentum_factor * diff
        

        # sorted_indices_before = torch.topk(similarity_before, dim=-1, k=similarity_before.shape[-1]).indices.cpu()
        # sorted_indices_after = torch.topk(similarity_after, dim=-1, k=similarity_after.shape[-1]).indices.cpu()

        # rank_blip = sorted_indices_before
        # rank_gpt = sorted_indices_after

        # # 转换第一个向量为字典
        # first_dict = {id: index for index, id in enumerate(rank_blip)}

        # # 创建前进和后退的列表
        # forward_ids = []
        # backward_ids = []

        # # 遍历第二个向量
        # for id in rank_gpt:
        #     if first_dict[id] > rank_gpt.index(id):
        #         backward_ids.append(id)
        #     elif first_dict[id] < rank_gpt.index(id):
        #         forward_ids.append(id)

    # Compute the similarity
    else:
        similarity = predicted_features @ index_features.T
    sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Generate prediction dicts
    queryid_to_retrieved_images = {query_id: query_sorted_names[:50].tolist() for
                                   (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)}

    return queryid_to_retrieved_images


args = args_define.args

def main():
    if args.eval_type in ['LDRE-B', 'LDRE-L', 'LDRE-G']:
        if args.eval_type == 'LDRE-B':
            clip_model_name = 'ViT-B/32'
        elif args.eval_type == 'LDRE-L':
            clip_model_name = 'ViT-L/14'
        else:
            clip_model_name = 'ViT-g-14'

        if clip_model_name == 'ViT-g-14':
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
        else:
            clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

        if args.preprocess_type == 'targetpad':
            print('Target pad preprocess pipeline is used')
            preprocess = targetpad_transform(1.25, 224)
        elif args.preprocess_type == 'clip':
            print('CLIP preprocess pipeline is used')
            preprocess = clip_preprocess
        else:
            raise ValueError("Preprocess type not supported")

        if args.dataset.lower() == 'cirr':
            relative_test_dataset = CIRRDataset(args.dataset_path, 'test', 'relative', preprocess, no_duplicates=True)
        elif args.dataset.lower() == 'circo':
            relative_test_dataset = CIRCODataset(args.dataset_path, 'test', 'relative', preprocess)
        else:
            raise ValueError("Dataset not supported")

        clip_model = clip_model.float().to(device)
        

    print(f"Eval type = {args.eval_type} \t exp name = {args.exp_name} \t")

    if args.dataset == 'cirr':
        cirr_generate_test_submission_file(args.dataset_path, clip_model_name, preprocess, args.submission_name)
    elif args.dataset == 'circo':
        circo_generate_test_submission_file(args.dataset_path, clip_model_name, preprocess, args.submission_name)

    else:
        raise ValueError("Dataset not supported")


if __name__ == '__main__':
    main()
