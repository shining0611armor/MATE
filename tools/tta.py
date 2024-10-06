from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
import datasets.tta_datasets as tta_datasets
from torch.utils.data import DataLoader
from utils.misc import *
level = [5]


def load_tta_dataset(args, config):
    # we have 3 choices - every tta_loader returns only point and labels
    root = config.tta_dataset_path  # being lazy - 1

    if args.dataset_name == 'modelnet':
        root = '/content/MATE/modelnet40_c/modelnet40_c'


        if args.corruption == 'clean':
            inference_dataset = tta_datasets.ModelNet_h5(args, root)
            tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
        else:
            inference_dataset = tta_datasets.ModelNet40C(args, root)
            tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    elif args.dataset_name == 'partnet':
        if args.corruption != 'clean':
            root = os.path.join(root, f'{args.dataset_name}_c',
                                f'{args.corruption}_{args.severity}')
        else:
            root = os.path.join(root, f'{args.dataset_name}_c',
                                f'{args.corruption}')

        inference_dataset = tta_datasets.PartNormalDataset(root=root, npoints=config.npoints, split='test',
                                                           normal_channel=config.normal, debug=args.debug)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
    elif args.dataset_name == 'scanobject':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ScanObjectNN(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    elif args.dataset_name == 'shapenetcore':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ShapeNetCore(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    else:
        raise NotImplementedError(f'TTA for {args.tta} is not implemented')

    print(f'\n\n Loading data from ::: {root} ::: level ::: {args.severity}\n\n')

    return tta_loader


def load_base_model(args, config, logger, load_part_seg=False):
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts, load_part_seg)
    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[
            args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    return base_model



def tta(args, config, train_writer=None):
    dataset_name = args.dataset_name
    npoints = config.npoints
    logger = get_logger(args.log_name)

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            acc_sliding_window = list()
            acc_avg = list()
            if args.corruption == 'clean':
                continue
                # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            if corr_id == 0:  # for saving results for easy copying to google sheet

                f_write, logtime = get_writer_to_all_result(args, config, custom_path='results_final_tta/')
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
                f_write.write(f'Corruption LEVEL: {args.severity}' + '\n\n')

            tta_loader = load_tta_dataset(args, config)
            total_batches = len(tta_loader)
            test_pred = []
            test_label = []
            if args.online:
                base_model = load_base_model(args, config, logger)
                optimizer = builder.build_opti_sche(base_model, config)[0]
                args.grad_steps = 1

            for idx, (data, labels) in enumerate(tta_loader):
                losses = AverageMeter(['Reconstruction Loss'])

                if not args.online:
                    base_model = load_base_model(args, config, logger)
                    optimizer = builder.build_opti_sche(base_model, config)[0]
                base_model.zero_grad()
                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:
                    pass

                # TTA Loop (for N grad steps)
                for grad_step in range(args.grad_steps):
                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'shapenetcore':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name in ['scanobject', 'scanobject_nbg']:
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    else:
                        raise NotImplementedError

                    # make a batch
                    if idx % args.stride_step == 0 or idx == len(tta_loader) - 1:
                        points = [points for _ in range(args.batch_size_tta)]
                        points = torch.squeeze(torch.vstack(points))

                        loss_recon, loss_p_consistency, loss_regularize = base_model(points)
                        loss = loss_recon + (args.alpha * loss_regularize)  # + (0.0001 * loss_p_consistency)
                        loss = loss.mean()
                        loss.backward()
                        optimizer.step()
                        base_model.zero_grad()
                        optimizer.zero_grad()
                    else:
                        continue

                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        losses.update([loss.item() * 1000])
                    else:
                        losses.update([loss.item() * 1000])

                    print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                              f'GradStep - {grad_step} / {args.grad_steps},'
                              f'Reconstruction Loss {[l for l in losses.val()]}',
                              logger=logger)

                # now inferring on this one sample
                base_model.eval()
                points = data.cuda()
                labels = labels.cuda()
                points = misc.fps(points, npoints)

                logits = base_model.module.classification_only(points, only_unmasked=False)
                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)

                test_pred.append(pred.detach())
                test_label.append(target.detach())

                if idx % 50 == 0:
                    test_pred_ = torch.cat(test_pred, dim=0)
                    test_label_ = torch.cat(test_label, dim=0)

                    if args.distributed:
                        test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                        test_label_ = dist_utils.gather_tensor(test_label_, args)

                    acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.

                    print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                              logger=logger)

                    acc_avg.append(acc.cpu())
            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                      logger=logger)
            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                f_write.close()

                print(f'Final Results Saved at:', os.path.join('results_final/', f'{logtime}_results.txt'))
                if train_writer is not None:
                    train_writer.close()

