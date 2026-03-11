import torch
from collections import Counter, OrderedDict
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm

from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel
import time
from basicsr.data.data_util import duf_downsample, generate_frame_indices, read_img_seq, read_npy_seq, read_img_seq_gray
import cv2
import numpy as np
from basicsr.utils import img2tensor, scandir, img2tensor_gray

# @MODEL_REGISTRY.register()
# class VideoRecurrentModel(VideoBaseModel):
#     def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
#         dataset = dataloader.dataset
#         dataset_name = dataset.opt['name']
#         with_metrics = self.opt['val']['metrics'] is not None
#         # initialize self.metric_results
#         # It is a dict: {
#         #    'folder1': tensor (num_frame x len(metrics)),
#         #    'folder2': tensor (num_frame x len(metrics))
#         # }
#         if with_metrics:
#             if not hasattr(self, 'metric_results'):  # only execute in the first run
#                 self.metric_results = {}
#                 num_frame_each_folder = Counter(dataset.data_info['folder'])
#                 for folder, num_frame in num_frame_each_folder.items():
#                     self.metric_results[folder] = torch.zeros(
#                         num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cpu')
#             # initialize the best metric results
#             self._initialize_best_metric_results(dataset_name)
#         # zero self.metric_results
#         rank, world_size = get_dist_info()
#         if with_metrics:
#             for _, tensor in self.metric_results.items():
#                 tensor.zero_()
#
#         metric_data = dict()
#         num_folders = len(dataset)
#         num_pad = (world_size - (num_folders % world_size)) % world_size
#         if rank == 0:
#             pbar = tqdm(total=len(dataset), unit='folder')
#         # Will evaluate (num_folders + num_pad) times, but only the first num_folders results will be recorded.
#         # (To avoid wait-dead)
#
#         #
#
#         if hasattr(self.net_g, 'module'):
#             net_g = self.net_g.module
#         else:
#             net_g = self.net_g
#
#         try:
#             net_g_cuda = net_g.cuda()
#         except Exception as e:
#             net_g_cuda = net_g
#
#         for i in range(rank, num_folders + num_pad, world_size):
#             idx = min(i, num_folders - 1)
#             val_data = dataset[idx]
#             folder = val_data['folder']
#             t, c, h, w = val_data['lq'].shape
#             hidden_states = None
#             visuals = {'result': [], 'gt': val_data['gt'].view(1, t, c, 4*h, 4*w)}
#             execution_all = 0.0
#             for fi in range(t):
#                 inp_cuda = val_data['lq'][fi:fi+1].cuda().view(1, 1, c, h, w)
#                 mv_cuda = val_data['mv'][fi:fi+1].cuda().view(1, 1, 2, h, w)
#                 res_cuda = val_data['res'][fi:fi+1].cuda().view(1, 1, 1, h, w)
#                 if fi % 25 == 0:
#                     hidden_states = None
#                 with torch.no_grad():
#                     start_time1 = time.time()
#                     gt_cuda, hidden_states = net_g_cuda(inp_cuda,mv_cuda,res_cuda,hidden_states, return_hs=True)
#                     end_time1 = time.time()
#                     if fi>=25:
#                         execution_time1 = end_time1 - start_time1
#                         execution_all = execution_all + execution_time1
#                 gt_cpu = gt_cuda.cpu()
#                 visuals['result'].append(gt_cpu)
#                 del gt_cuda, inp_cuda, mv_cuda
#                 torch.cuda.empty_cache()
#             execution_mean = execution_all*1000/75
#             print(f"平均每帧运行时间：{execution_mean:.2f}ms (循环内计算)")
#             visuals['result'] = torch.cat(visuals['result'], dim=1)
#
#             # evaluate
#             if i < num_folders:
#                 for idx in range(visuals['result'].size(1)):
#                     result = visuals['result'][0, idx, :, :, :]
#                     result_img = tensor2img([result])  # uint8, bgr
#                     metric_data['img'] = result_img
#                     if 'gt' in visuals:
#                         gt = visuals['gt'][0, idx, :, :, :]
#                         gt_img = tensor2img([gt])  # uint8, bgr
#                         metric_data['img2'] = gt_img
#
#                     if save_img:
#                         if self.opt['is_train']:
#                             raise NotImplementedError('saving image is not supported during training.')
#                         else:
#                             # if self.center_frame_only:  # vimeo-90k
#                             if False:
#                                 clip_ = val_data['lq_path'].split('/')[-3]
#                                 seq_ = val_data['lq_path'].split('/')[-2]
#                                 name_ = f'{clip_}_{seq_}'
#                                 img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
#                                                     f"{name_}_{self.opt['name']}.png")
#                             else:  # others
#                                 img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
#                                                     f"{idx:08d}_{self.opt['name']}.png")
#                             # image name only for REDS dataset
#                         imwrite(result_img, img_path)
#
#                     # calculate metrics
#                     if with_metrics:
#                         for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
#                             result = calculate_metric(metric_data, opt_)
#                             self.metric_results[folder][idx, metric_idx] += result
#
#                 # progress bar
#                 if rank == 0:
#                     for _ in range(world_size):
#                         pbar.update(1)
#                         pbar.set_description(f'Folder: {folder}')
#
#         if rank == 0:
#             pbar.close()
#
#         if with_metrics:
#             if self.opt['dist']:
#                 # collect data among GPUs
#                 for _, tensor in self.metric_results.items():
#                     dist.reduce(tensor, 0)
#                 dist.barrier()
#
#             if rank == 0:
#                 self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
#
#     def test(self):
#         # print(self.lq.shape)
#         n = self.lq.size(1)
#         self.net_g.eval()
#
#         flip_seq = self.opt['val'].get('flip_seq', False)
#         self.center_frame_only = self.opt['val'].get('center_frame_only', False)
#
#         if flip_seq:
#             self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)
#
#         with torch.no_grad():
#             print('test','lq:',self.lq.shape,'mv:',self.mv.shape)
#             self.output = self.net_g(self.lq,self.mv)
#
#         if flip_seq:
#             output_1 = self.output[:, :n, :, :, :]
#             output_2 = self.output[:, n:, :, :, :].flip(1)
#             self.output = 0.5 * (output_1 + output_2)
#
#         if self.center_frame_only:
#             self.output = self.output[:, n // 2, :, :, :]
#
#         self.net_g.train()
#
#     def optimize_parameters(self, current_iter):
#         self.optimizer_g.zero_grad()
#         self.output = self.net_g(self.lq,self.mv,self.res)
#
#         l_total = 0
#         loss_dict = OrderedDict()
#         # pixel loss
#         if self.cri_pix:
#             # l_pix = self.cri_pix(self.output[:, 1:], self.gt[:, 1:])
#             l_pix = self.cri_pix(self.output, self.gt)
#             # l_pix_I = self.cri_pix(self.output[:, 0], self.gt[:, 0])
#             l_total = l_pix
#             # l_total = l_pix_I
#             loss_dict['l_pix'] = l_pix
#             # loss_dict['l_pix_I'] = l_pix_I
#
#
#         # perceptual loss
#         if self.cri_perceptual:
#             l_percep, l_style = self.cri_perceptual(self.output, self.gt)
#             if l_percep is not None:
#                 l_total += l_percep
#                 loss_dict['l_percep'] = l_percep
#             if l_style is not None:
#                 l_total += l_style
#                 loss_dict['l_style'] = l_style
#
#         l_total.backward()
#         self.optimizer_g.step()
#
#         self.log_dict = self.reduce_loss_dict(loss_dict)
#
#         if self.ema_decay > 0:
#             self.model_ema(decay=self.ema_decay)


@MODEL_REGISTRY.register()
class VideoRecurrentModel(VideoBaseModel):
    def __init__(self, opt):
        super(VideoRecurrentModel, self).__init__(opt)
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        super(VideoRecurrentModel, self).optimize_parameters(current_iter)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {}
                num_frame_each_folder = Counter(dataset.data_info['folder'])
                for folder, num_frame in num_frame_each_folder.items():
                    self.metric_results[folder] = torch.zeros(
                        num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cpu')
            # initialize the best metric results
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        metric_data = dict()
        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')
        # Will evaluate (num_folders + num_pad) times, but only the first num_folders results will be recorded.
        # (To avoid wait-dead)

        #

        if hasattr(self.net_g, 'module'):
            net_g = self.net_g.module
        else:
            net_g = self.net_g

        try:
            net_g_cuda = net_g.cuda()
            # net_g_cuda = net_g.cpu()
        except Exception as e:
            net_g_cuda = net_g

        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']
            val_data_lq = cv2.imread(val_data['lq'][0]).astype(np.float32) / 255.
            val_data_lq = img2tensor(val_data_lq, bgr2rgb=True, float32=True)
            c, h, w = val_data_lq.shape
            t = 100
            hidden_states = None
            # visuals = {'result': [], 'gt': val_data['gt'].view(1, t, c, 4*h, 4*w)}
            execution_all = 0.0
            for fi in range(t):
                val_data_lq = cv2.imread(val_data['lq'][fi]).astype(np.float32) / 255.0
                val_data_lq = img2tensor(val_data_lq, bgr2rgb=True, float32=True)
                inp_cuda = val_data_lq.cuda().view(1, 1, c, h, w)

                val_data_mv = np.load(val_data['mv'][fi]).astype(np.float32)
                val_data_mv = torch.tensor(val_data_mv)
                mv_cuda = val_data_mv.cuda().view(1, 1, 2, h, w)

                val_data_res = np.load(val_data['res'][fi]).astype(np.float32) / 255.0
                val_data_res = torch.tensor(val_data_res)
                res_cuda = val_data_res.cuda().view(1, 1, 1, h, w)

                if fi % 25 == 0:
                    hidden_states = None
                with torch.no_grad():
                    torch.cuda.synchronize()
                    start_time1 = time.time()
                    gt_cuda, hidden_states = net_g_cuda(inp_cuda,mv_cuda,res_cuda,hidden_states, return_hs=True)
                    torch.cuda.synchronize()
                    end_time1 = time.time()
                    if fi>=25:
                        execution_time1 = end_time1 - start_time1
                        execution_all = execution_all + execution_time1
                gt_cpu = gt_cuda.cpu()

            # evaluate
                result = gt_cpu
                result_img = tensor2img([result])  # uint8, bgr
                metric_data['img'] = result_img

                val_data_gt = cv2.imread(val_data['gt'][fi]).astype(np.float32) / 255.
                val_data_gt = img2tensor(val_data_gt, bgr2rgb=True, float32=True)
                gt = val_data_gt.view(1, 1, c, 4*h, 4*w)
                gt_img = tensor2img([gt])  # uint8, bgr
                metric_data['img2'] = gt_img

                if save_img:
                    if self.opt['is_train']:
                        raise NotImplementedError('saving image is not supported during training.')
                    else:
                        # if self.center_frame_only:  # vimeo-90k
                        if False:
                            clip_ = val_data['lq_path'].split('/')[-3]
                            seq_ = val_data['lq_path'].split('/')[-2]
                            name_ = f'{clip_}_{seq_}'
                            img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                f"{name_}_{self.opt['name']}.png")
                        else:  # others
                            img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                f"{fi:08d}_{self.opt['name']}.png")

                        # image name only for REDS dataset
                    imwrite(result_img, img_path)

                # calculate metrics
                if with_metrics:
                    for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                        result = calculate_metric(metric_data, opt_)
                        self.metric_results[folder][idx, metric_idx] += result

            # progress bar
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Folder: {folder}')

            execution_mean = execution_all * 1000 / 75
            print(f"平均每帧运行时间：{execution_mean:.2f}ms")

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)



    def test(self):
        # print(self.lq.shape)
        n = self.lq.size(1)
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            print('test','lq:',self.lq.shape,'mv:',self.mv.shape)
            self.output = self.net_g(self.lq,self.mv)

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()
