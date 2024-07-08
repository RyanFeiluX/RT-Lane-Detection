import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
# from data.videoloader import load_video, get_frameprops
from PIL import Image


def infer_imgonline(vis, imgs):
    """
    Infer the images and determine lane sign.
    @param vis: ndrray h*w*c
    @param imgs: tensor b*c*h*w
    @return:
    """
    imgs = imgs.cuda()
    with torch.no_grad():
        out = net(imgs)

    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    out_j = out[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(cfg.griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == cfg.griding_num] = 0
    out_j = loc

    # import pdb; pdb.set_trace()
    # vis = cv2.imread(os.path.join(cfg.data_root, names[0]))
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                           int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                    cv2.circle(vis, ppp, 5, (0, 255, 0), -1)

    return vis


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    elif cfg.dataset == 'Custom':
        if args.test_cfg == 'CULane':
            cls_num_per_lane = 18
        elif args.test_cfg == 'Tusimple':
            cls_num_per_lane = 56
        else:
            raise RuntimeError('Invalid argument --test-cfg: %s' % args.test_cfg)
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if cfg.dataset == 'CULane':
        if args.test_list:
            splits = [f.strip() for f in args.test_list.split(',')]
        else:
            splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list/test_split/'+split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor
    elif cfg.dataset == 'Custom':
        splits = []
        datasets = []
        if args.test_cfg == 'CULane':
            row_anchor = culane_row_anchor
        elif args.test_cfg == 'Tusimple':
            row_anchor = tusimple_row_anchor
        else:
            raise RuntimeError('Invalid argument --test-cfg: %s' % args.test_cfg)
    else:
        raise NotImplementedError
    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(split[:-3]+'avi')
        vout = cv2.VideoWriter(split[:-3]+'avi', fourcc , 30.0, (img_w, img_h))
        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                out = net(imgs)

            col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
            col_sample_w = col_sample[1] - col_sample[0]


            out_j = out[0].data.cpu().numpy()
            out_j = out_j[:, ::-1, :]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(cfg.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == cfg.griding_num] = 0
            out_j = loc

            # import pdb; pdb.set_trace()
            vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                            cv2.circle(vis,ppp,5,(0,255,0),-1)
            vout.write(vis)
        vout.release()

    if cfg.dataset == 'Custom':
        from data.videoloader import CustomLoader
        vl = CustomLoader(cfg.crop)
        fcnt, fw, fh = vl.get_frameprops(args.video_in)
        nrfrm = int(fcnt / args.frame_interval)
        iter_frame = vl.load_video(args.video_in, frame_interval=args.frame_interval)
        img_w, img_h = fw, fh
        if args.video_out:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            assert os.path.exists(args.video_out), 'Directory %s does not exist' % args.video_out
            vout_path = os.path.join(args.video_out, os.path.splitext(os.path.basename(args.video_in))[0] + '_label.avi')
            print(vout_path, flush=True)
            vout = cv2.VideoWriter(str(vout_path), fourcc, 30.0, (img_w, img_h))
        for _ in tqdm.tqdm(range(nrfrm)):
            vis = iter_frame.__next__()  # return: ndarray h*w*c
            img = Image.fromarray(vis.copy()) #(cv2.cvtColor(vis.copy(), cv2.COLOR_BGR2RGB))  # return: Image
            img2 = img_transforms(img)  # return: tensor c*h*w
            vis2 = infer_imgonline(vis, img2[None, ...])  # return: ndaaray h*w*c
            if args.video_out:
                vout.write(vis2)

            cv2.imshow("Labeled lanes", vis2)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        iter_frame.close()
        # cv2.destroyAllWindows()
        if args.video_out:
            print('%s is created' % vout_path)
        vout.release()
