import csv
import os
from pathlib import Path
from timeit import default_timer as timer

import torch
import torch.nn.parallel
from easydict import EasyDict
from torch.nn import functional as F

import utils
from loss import ContrastiveLoss, CycleConsistencyLoss
from model import CootModel
from optimizer import get_optimizer, ReduceLROnPlateauWarmup


def unpack_data(data_dict, use_cuda):
    def to_device(x):
        if use_cuda and isinstance(x, torch.Tensor):
            return x.cuda(non_blocking=True)
        return x

    return [
        to_device(data_dict[a]) for a in
        ("vid_id", "vid_frames", "vid_frames_mask", "vid_frames_len",
         "par_cap_vectors", "par_cap_mask", "par_cap_len",
         "clip_num", "clip_frames", "clip_frames_len", "clip_frames_mask",
         "sent_num", "sent_cap_vectors", "sent_cap_mask", "sent_cap_len")]


class TrainerVideoText:
    def __init__(self, log_dir: str, cfg: EasyDict, use_cuda: bool = True,
                 use_multi_gpu: bool = True, load_ckpt: str = "",
                 is_train=True):
        self.use_multi_gpu = use_multi_gpu
        self.use_cuda = use_cuda
        self.cfg = cfg
        self.log_dir = Path(log_dir)

        self.timer_start_train = 0
        self.det_best_field_current = 0
        self.det_best_field_best = 0
        self.best_epoch = 0

        # logger / metrics
        self.metrics_fh = None
        if is_train:
            os.makedirs(self.log_dir, exist_ok=True)
            metrics_file = self.log_dir / "train_metrics.csv"
            metric_keys = utils.get_csv_header_keys(
                cfg.training.compute_clip_retrieval)
            self.metrics_fh = metrics_file.open("wt", encoding="utf8")
            self.metrics_writer = csv.DictWriter(self.metrics_fh, metric_keys)
            self.metrics_writer.writeheader()
            self.metrics_fh.flush()
            utils.dump_config(cfg, self.log_dir / "config.yaml")
        self.logger = utils.get_logger(
            self.log_dir, "trainer", log_file=is_train)
        # model
        self.model = CootModel(cfg, use_cuda, use_multi_gpu)

        # contrastive loss
        self.loss_f_contr = ContrastiveLoss(use_cuda)

        # cycle consistency loss
        self.loss_f_cyclecons = None
        if cfg.training.loss_cycle_cons != 0:
            self.loss_f_cyclecons = CycleConsistencyLoss(
                num_samples=1, use_cuda=use_cuda)

        # optimizer
        self.optimizer = get_optimizer(cfg.optimizer, self.model.get_params())

        # scheduler
        self.lr_scheduler = ReduceLROnPlateauWarmup(
            self.optimizer, cfg.scheduler.warmup, mode="max",
            patience=cfg.scheduler.patience, cooldown=cfg.scheduler.cooldown)

        if load_ckpt != "":
            self.logger.info(f"Load checkpoint {load_ckpt}")
            self.model.load_checkpoint(load_ckpt)

    def compare_metrics(self, comparison, best):
        if best is None:
            return True
        threshold = 1e-4
        rel_epsilon = threshold + 1.
        return comparison > best * rel_epsilon

    def compute_align_loss(self, v_embedding, p_embedding):
        return self.loss_f_contr(v_embedding, p_embedding)

    def compute_cluster_loss(self, v_embedding, p_embedding):
        return (self.loss_f_contr(v_embedding, v_embedding)
                + self.loss_f_contr(p_embedding, p_embedding)) / 2

    def compute_total_constrastive_loss(
            self, vid_emb, par_emb, clip_emb, sent_emb, vid_context,
            par_context):
        vid_context_norm = F.normalize(vid_context)
        clip_emb_norm = F.normalize(clip_emb)
        vid_emb_norm = F.normalize(vid_emb)
        par_context_norm = F.normalize(par_context)
        sent_emb_norm = F.normalize(sent_emb)
        par_emb_norm = F.normalize(par_emb)
        loss = self.compute_align_loss(vid_emb_norm, par_emb_norm)
        loss += self.compute_align_loss(clip_emb_norm, sent_emb_norm)
        loss += self.compute_align_loss(vid_context_norm, par_context_norm)
        loss += self.compute_cluster_loss(vid_emb_norm, par_emb_norm)
        loss += self.compute_cluster_loss(clip_emb_norm, sent_emb_norm)
        return loss

    def compute_cmc_loss(
            self, clip_emb_reshape, clip_emb_mask, clip_emb_lens,
            sent_emb_reshape, sent_emb_mask, sent_emb_lens):
        if self.loss_f_cyclecons is not None:
            clip_clip_loss, sent_sent_loss = self.loss_f_cyclecons(
                clip_emb_reshape, clip_emb_mask, clip_emb_lens,
                sent_emb_reshape, sent_emb_mask, sent_emb_lens)
            loss = self.cfg.training.loss_cycle_cons * (
                    clip_clip_loss + sent_sent_loss)
            return loss
        return 0

    def close(self):
        if self.metrics_fh is not None:
            self.metrics_fh.close()
        utils.close_logger(self.logger)

    def train_loop(self, train_loader, val_loader):
        max_step = len(train_loader)
        self.timer_start_train = timer()
        epoch = 0
        for epoch in range(0, self.cfg.training.num_epochs):
            self.model.train()

            # train one epoch
            self.logger.info(
                "---------- Training epoch {} ----------".format(
                    epoch))
            for step, data_dict in enumerate(train_loader):
                (vid_id, vid_frames, vid_frames_mask, vid_frames_len,
                 par_cap_vectors, par_cap_mask, par_cap_len,
                 clip_num, clip_frames, clip_frames_len, clip_frames_mask,
                 sent_num, sent_cap_vectors, sent_cap_mask,
                 sent_cap_len) = unpack_data(data_dict, self.use_cuda)

                # forward pass
                (vid_emb, clip_emb, vid_context, clip_emb_reshape,
                 clip_emb_mask, clip_emb_lens) = self.model.encode_video(
                    vid_frames, vid_frames_mask, vid_frames_len,
                    clip_num, clip_frames, clip_frames_len, clip_frames_mask)
                (par_emb, sent_emb, par_context, sent_emb_reshape,
                 sent_emb_mask, sent_emb_lens) = self.model.encode_paragraph(
                    par_cap_vectors, par_cap_mask, par_cap_len,
                    sent_num, sent_cap_vectors, sent_cap_mask, sent_cap_len)
                loss = self.compute_total_constrastive_loss(
                    vid_emb, par_emb, clip_emb, sent_emb, vid_context,
                    par_context)
                loss += self.compute_cmc_loss(
                    clip_emb_reshape, clip_emb_mask, clip_emb_lens,
                    sent_emb_reshape, sent_emb_mask, sent_emb_lens)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # logging
                if step % 10 == 0:
                    el_time = (timer() - self.timer_start_train) / 60
                    l_ms = len(str(max_step))
                    str_step = ("{:" + str(l_ms) + "d}").format(step)
                    print_string = (
                        f"E{epoch}[{str_step}/{max_step}] T {el_time:.3f}m "
                        f"LR {self.optimizer.param_groups[0]['lr']:5.3e} "
                        f"L {loss:.4f}")
                    self.logger.info(print_string)

            # validate one epoch
            self.logger.info(
                "---------- Validating epoch {} ----------".format(epoch))
            vid_metrics, clip_metrics = self.validate(val_loader)
            v2p_res, p2v_res, vid_best_at_1 = vid_metrics
            c2s_res, s2c_res, clip_best_at_1 = None, None, None
            if clip_metrics is not None:
                c2s_res, s2c_res, clip_best_at_1 = clip_metrics

            # find field which determines is_best
            if self.cfg.training.det_best_field == "val_score_at_1":
                self.det_best_field_current = vid_best_at_1
            elif self.cfg.training.det_best_field == "val_clip_score_at_1":
                self.det_best_field_current = clip_best_at_1
            else:
                raise NotImplementedError

            # check if best
            is_best = self.compare_metrics(
                self.det_best_field_current, self.det_best_field_best)
            if is_best:
                self.det_best_field_best = self.det_best_field_current
                self.best_epoch = epoch

            # write validation results to csv
            csv_input = {
                "ep": epoch,
                "time": timer() - self.timer_start_train
            }
            for key_ret, dict_ret in zip(
                    ["v", "p", "c", "s"],
                    [v2p_res, p2v_res, c2s_res, s2c_res]):
                if dict_ret is None:
                    continue
                for key in utils.EVALKEYS:
                    csv_input.update([(f"{key_ret}-{key}", dict_ret[key])])
            self.metrics_writer.writerow(csv_input)
            self.metrics_fh.flush()

            # step lr scheduler
            self.lr_scheduler.step_rop(
                self.det_best_field_current, True)
            self.logger.info(
                f"ROP: model improved: {is_best}, "
                f"value {self.det_best_field_current:.3f},"
                f"new LR: {self.optimizer.param_groups[0]['lr']:5.3e}")

            # save checkpoint
            self.model.save_checkpoint(self.log_dir / f"ckpt_ep{epoch}.pth")

            # check if model did not improve for too long
            term_after = 15
            if epoch - self.best_epoch > term_after:
                self.logger.info(
                    f"NO improvements for {term_after} epochs (current "
                    f"{epoch} best {self.best_epoch}) STOP training.")
                break

        time_total = timer() - self.timer_start_train
        self.logger.info(
            "Training {} epochs took {:.3f}s / {:.3f}s/ep val".format(
                epoch, time_total, time_total / epoch))

    def validate(self, val_loader, debug_max=-1):
        self.model.eval()
        max_step = len(val_loader)
        do_clip_ret = self.cfg.training.compute_clip_retrieval

        # collect embeddings
        vid_emb_list = []
        par_emb_list = []
        clip_emb_list = []
        sent_emb_list = []
        for step, data_dict in enumerate(val_loader):
            if step >= debug_max > -1:
                break
            (vid_id, vid_frames, vid_frames_mask, vid_frames_len,
             par_cap_vectors, par_cap_mask, par_cap_len,
             clip_num, clip_frames, clip_frames_len, clip_frames_mask,
             sent_num, sent_cap_vectors, sent_cap_mask,
             sent_cap_len) = unpack_data(data_dict, self.use_cuda)
            if step == 0:
                print(f"ids {vid_id[:4]}...")
            # forward pass
            (vid_emb, clip_emb, vid_context, clip_emb_reshape,
             clip_emb_mask, clip_emb_lens) = self.model.encode_video(
                vid_frames, vid_frames_mask, vid_frames_len,
                clip_num, clip_frames, clip_frames_len, clip_frames_mask)
            (par_emb, sent_emb, par_context, sent_emb_reshape,
             sent_emb_mask, sent_emb_lens) = self.model.encode_paragraph(
                par_cap_vectors, par_cap_mask, par_cap_len,
                sent_num, sent_cap_vectors, sent_cap_mask, sent_cap_len)
            loss = self.compute_total_constrastive_loss(
                vid_emb, par_emb, clip_emb, sent_emb, vid_context,
                par_context)
            # loss += self.compute_cmc_loss(
            #     clip_emb_reshape, clip_emb_mask, clip_emb_lens,
            #     sent_emb_reshape, sent_emb_mask, sent_emb_lens)
            # collect embeddings
            vid_emb_list.extend(vid_emb.detach().cpu())
            par_emb_list.extend(par_emb.detach().cpu())
            if do_clip_ret:
                clip_emb_list.extend(clip_emb.detach().cpu())
                sent_emb_list.extend(sent_emb.detach().cpu())
            # logging
            if step % 10 == 0:
                self.logger.info(
                    f"Val [{step}/{max_step}] Loss {loss.item():.4f}")
        vid_emb_list = torch.stack(vid_emb_list, 0)
        par_emb_list = torch.stack(par_emb_list, 0)
        if do_clip_ret:
            clip_emb_list = torch.stack(clip_emb_list, 0)
            sent_emb_list = torch.stack(sent_emb_list, 0)
        # video text retrieval
        vid_emb_list = F.normalize(vid_emb_list).numpy()
        par_emb_list = F.normalize(par_emb_list).numpy()
        # np.save("emb_vid.npy", vid_emb_list)
        # np.save("emb_par.npy", par_emb_list)
        v2p_res, v2p_top1, v2p_ranks = utils.compute_retr_vid_to_par(
            vid_emb_list, par_emb_list)
        p2v_res, p2v_top1, p2v_ranks = utils.compute_retr_par_to_vid(
            vid_emb_list, par_emb_list)
        sum_at_1 = v2p_res["r1"] + p2v_res["r1"]
        self.logger.info(utils.EVALHEADER)
        self.logger.info(utils.retrieval_results_to_str(p2v_res, "Par2Vid"))
        self.logger.info(utils.retrieval_results_to_str(v2p_res, "Vid2Par"))
        self.logger.info(f"Retrieval done: {self.log_dir} "
                         f"{len(vid_emb_list)} Items.")
        if not do_clip_ret:
            return (v2p_res, p2v_res, sum_at_1), None
        # clip sentence retrieval
        clip_emb_list = F.normalize(clip_emb_list).numpy()
        sent_emb_list = F.normalize(sent_emb_list).numpy()
        c2s_res, c2s_top1, c2s_ranks = utils.compute_retr_vid_to_par(
            clip_emb_list, sent_emb_list)
        s2c_res, s2c_top1, s2c_ranks = utils.compute_retr_par_to_vid(
            clip_emb_list, sent_emb_list)
        c2s_sum_at_1 = c2s_res["r1"] + s2c_res["r1"]
        self.logger.info(utils.EVALHEADER)
        self.logger.info(utils.retrieval_results_to_str(s2c_res, "Sen2Cli"))
        self.logger.info(utils.retrieval_results_to_str(c2s_res, "Cli2Sen"))
        self.logger.info(f"{len(clip_emb_list)} Items.")
        return ((v2p_res, p2v_res, sum_at_1),
                (c2s_res, s2c_res, c2s_sum_at_1))
