import os
import argparse
import random
import time
import datetime

import torch
import numpy as np
import wandb
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

from data.dataset import build_dataset_single
from model.flair_distill_flair import FLAIRMultiLayer
from process.finetune_atp import train_one_epoch, evaluate
from utils.eval import save_model
import warnings

from retfound.models_vit import RETFound_mae  


warnings.filterwarnings("ignore", category=FutureWarning, module="timm.layers")
def get_args_parser():
    parser = argparse.ArgumentParser('EyeCoOp', add_help=False)
    parser.add_argument('--modality', default='fundus', type=str, help='modality for training backbone model')
    parser.add_argument('--device_id', default='0', type=str, help='select device id')
    parser.add_argument('--device', default='cuda', type=str, help='device: cuda or cpu')
    
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                    help='epochs to warmup LR')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    
    parser.add_argument('--data_path', default='Your Dataset Path', type=str,help='dataset path')
    parser.add_argument('--concept_path', default='Your Concept Path', type=str, help='concept path')
    
    # Augmentation parameters3
    parser.add_argument('--input_size', default=512, type=int,
                    help='images input size')
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_samples', default=36000, type=int, help='number of the sampled training data')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--n_classes', default=9, type=int, help='number of the classification types')
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--accum_iter', default=2, type=int)
    
    parser.add_argument('--print_freq', default=100, type=int, help='batch size')

    parser.add_argument('--eval', action='store_true', default=False, help='Perform evaluation only')
    parser.add_argument('--output_dir', default='./checkpoints/', help='path where to save, empty for no saving')

    parser.add_argument('--lambda_sccm', type=float, default=0.1)
    parser.add_argument('--lambda_kdsp', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=8.0)  
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--use_atprompt', type=bool,default=True,
                        help='Whether to use attribute words in text prompts')
    parser.add_argument('--att_num', type=int, default=2,
                        help='How many attribute groups to use (1 or 2)')
    parser.add_argument('--n_att1', type=int, default=2,
                        help='number of learnable tokens for attribute 1')
    parser.add_argument('--att1_text', type=str, default='Location',
                        help='attribute word/phrase 1')
    parser.add_argument('--n_att2', type=int, default=2,
                        help='number of learnable tokens for attribute 2')
    parser.add_argument('--att2_text', type=str, default='Morphology',
                        help='attribute word/phrase 2')
    parser.add_argument('--use_retfound_teacher', type=bool,default=True,
                        help='whether to use RETFound ViT as image teacher')
    parser.add_argument('--retfound_ckpt', type=str, default='Your RETFound Checkpoint Path',
                        help='path to RETFound checkpoint (e.g. RETFound_mae.pth)')
    parser.add_argument('--teacher_dim', type=int, default=1024,
                        help='feature dim of RETFound encoder (default 1024)')
    return parser

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device, int(args.device_id))

    torch.backends.cudnn.benchmark = True
    
    train_dataset = build_dataset_single('train', args=args, mod=args.modality)
    dev_dataset = build_dataset_single('dev', args=args, mod=args.modality)
    test_dataset = build_dataset_single('test', args=args, mod=args.modality)

    sampler_train = None
    cls_count = train_dataset.label_statistics().astype(np.float32)
    print("cls_count:", cls_count)

    num_classes = args.n_classes
    assert len(cls_count) == num_classes, "cls_count different from n_classes "

    N = cls_count.sum()
    K = float(num_classes)
    class_weights = N / (K * cls_count)
    print("raw class_weights:", class_weights)
    class_weights = np.sqrt(class_weights)           
    max_w = 10.0
    class_weights = np.clip(class_weights, 1.0, max_w)

    print("smoothed class_weights:", class_weights)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)


    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=True,
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dev_dataset, 
        batch_size=args.batch_size, 
        pin_memory=args.pin_mem, 
        shuffle=False, num_workers=args.num_workers,
        drop_last=False)

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        pin_memory=args.pin_mem, 
        shuffle=False, num_workers=args.num_workers,
        drop_last=False)
    
    concept_feat_path = os.path.join(args.concept_path, 'concepts_raw.npy')
    args.classnames = ['NOR','AMD','CSC','DR','GLC','MEM','MYO','RVO','WAMD']
    model = FLAIRMultiLayer(args, device, concept_feat_path, class_weights=class_weights_tensor)

    ######===============CLIP============================###########
    # for p in model.clip.parameters():
    #     p.requires_grad = False
    # for p in model.image_encoder.parameters():
    #     p.requires_grad = True
    # for p in model.prompt_learner.parameters():
    #     p.requires_grad = True
    # for p in model.concept_classifier.parameters():
    #     p.requires_grad = False
    # model.project_4.requires_grad = False
    ##############==================clip==================########

    #################=========Flair============================###########
    for p in model.flair_model.parameters():
        p.requires_grad = False
    for p in model.flair_model.vision_model.parameters():
        p.requires_grad = True
    for p in model.prompt_learner.parameters():
        p.requires_grad = True
    for p in model.concept_classifier.parameters():
        p.requires_grad = False
    for p in model.teacher_proj.parameters():
        p.requires_grad = False
    #################=========Flair============================###########

    model = model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model parameters = %s' % str(n_parameters))
    teacher_model = None
    if args.use_retfound_teacher:
        print("=> Build RETFound teacher ...")
        teacher_img_size = 224  
        teacher_model = RETFound_mae(
            img_size=teacher_img_size,     
            num_classes=args.n_classes,
            global_pool=True,
        )
        if args.retfound_ckpt:
            ckpt = torch.load(args.retfound_ckpt, map_location='cpu')
            state = ckpt.get('model', ckpt)
            msg = teacher_model.load_state_dict(state, strict=False)
            print("[RetFound] load_state_dict:", msg)
        teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
    else:
        print("=> Not using RetFound teacher (use_retfound_teacher=False)")

    optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=args.lr,
    weight_decay=args.weight_decay
    )
    print("optimizer = %s" % str(optimizer))

    if args.eval:
        test_stats, test_metric = evaluate(args, data_loader_test, model, device)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_metric = 0.0

    best_test_metric = 0.0
    cw_start = 80
    cw_end   = 150
    if cw_end <= cw_start:
        cw_end = cw_start + 1
    print(f"[main] class_weight will start annealing from epoch {cw_start} to {cw_end}")

    for epoch in range(args.epochs):
        if hasattr(model, "set_ce_weight_factor") and getattr(model, "base_class_weights", None) is not None:
            if epoch <= cw_start:
                alpha = 1.0
            elif epoch >= cw_end:
                alpha = 0.0
            else:
               
                alpha = 1.0 - (epoch - cw_start) / float(cw_end - cw_start)
            model.set_ce_weight_factor(alpha)
            if epoch % 10 == 0 or epoch in (cw_start, cw_end):
                print(f"[epoch {epoch}] ce_weight_factor = {alpha:.4f}")

        train_stats = train_one_epoch(
            model,data_loader_train,
            optimizer, device, epoch,
            args, teacher_model)

        val_stats, val_metric = evaluate(args, data_loader_val, model, device)

        if max_metric < val_metric:
            max_metric = val_metric
            if args.modality == 'fundus':
                test_stats, test_metric = evaluate(args, data_loader_test, model, device)
            else:
                test_metric = 0.0   
            if test_metric > best_test_metric:
                best_test_metric = test_metric

                if args.output_dir:
                    save_model(args=args, model=model, optimizer=optimizer, epoch=epoch, if_best=True)
                print(f'------ best model ------  (val = {max_metric:.4f}, test = {best_test_metric:.4f})')
            

        if epoch==(args.epochs-1):
            print('------ last model ------')
            if args.modality == 'fundus':
                test_stats, test_metric = evaluate(args, data_loader_test, model, device)
            if args.output_dir:
                save_model(args=args, model=model, optimizer=optimizer, epoch=epoch, if_best=False)
        
        wandb_log = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'val_{k}': v for k, v in val_stats.items()}}
        wandb.log(wandb_log)

                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    wandb.init(
        config={
            "modality": args.modality,
            "learning_rate": args.lr,
            "output_dir": args.output_dir,
            "num_classes": args.n_classes,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "num_epochs": args.epochs,
        }
    )
    
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
    