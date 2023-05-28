import torchvision.models
import pytorch_lightning as pl
from pytorch_metric_learning.losses import SelfSupervisedLoss
from torchvision import transforms as tfm
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
import parser
from modules.ReweightVLAD import ReweightVLAD
from utils import compute_recalls
from modules.TI import TemplateInjector
from modules.MixVPR import *
from modules.GeM import GeM
from modules.GPM import *
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
import os

torch.cuda.empty_cache()

torch.set_float32_matmul_precision("highest")
torch.cuda.set_per_process_memory_fraction(1 / 3, torch.cuda.current_device())  # Use only 1/3 of the available memory

s = 32
dev = torch.device('cuda')
torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


class GeoModel(pl.LightningModule):
    def __init__(self, val_dataset, test_dataset, descriptors_dim=512, num_preds_to_save=0, save_only_wrong_preds=True,
                 proxy_bank: ProxyBank = None, proxy_head: ProxyHead = None,
                 mix: MixVPR = None):
        super().__init__()
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_preds_to_save = num_preds_to_save
        self.save_only_wrong_preds = save_only_wrong_preds

        # Use a pretrained model
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

        # Change the output of the FC layer to the desired descriptors dimension
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, descriptors_dim)
        self.model.avgpool = GeM()

        # Set the loss function
        self.loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        self.save_hyperparameters(ignore=['proxy_head'])

        # Instantiate the Proxy Head and Proxy Bank
        if args.gpm:
            self.phead = proxy_head
            self.pbank = proxy_bank
        if args.feature_mixing:
            # It dodes not pass over the constructor
            self.model = nn.Sequential(*self.model.children()[:-2], mix)
        if args.template_injection:
            self.ti = TemplateInjector(224)
            self.loss_fn = SelfSupervisedLoss(self.loss_fn)
            self.margin = 2
        if args.reweighting:
            self.model = nn.Sequential(*self.model[-2])  # convolutional part
            self.reweighting = ReweightVLAD(dim=49, alpha=75)

    def forward(self, images):
        if args.reweighting and args.template_injection:
            template_descriptors = self.ti(images)
            template_descriptors = self.model(template_descriptors)
            descriptors = self.model(images)
            descriptors = self.reweighting(descriptors, template_descriptors)
        elif args.template_injection:
            template_descriptors = self.ti(images)
            descriptors = self.model(template_descriptors)
        elif args.reweighting:
            descriptors = self.model(images)
            descriptors = self.reweighting(descriptors)
        else:
            descriptors = self.model(images)

        return descriptors

    def configure_optimizers(self):
        optimizers = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=0.001, momentum=0.9)
        return optimizers

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        loss = self.loss_fn(descriptors, labels)
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        images, labels = batch
        num_places, num_images_per_place, C, H, W = images.shape
        images = images.view(num_places * num_images_per_place, C, H, W)
        descriptors = self(images)
        labels = labels.view(num_places * num_images_per_place)

        if args.template_injection:
            template_descriptors = self.ti(images)
            template_descriptors = self(template_descriptors)
            template_distance = torch.norm(descriptors - template_descriptors, p=2, dim=1)
            # loss = loss + torch.max(torch.zeros_like(template_distance), self.margin - template_distance).mean()
            loss = self.loss_function(descriptors, template_descriptors)
        else:
            loss = self.loss_function(descriptors, labels)  # Call the loss_function we defined above
        # Feed forward the batch to the model
        if args.gpm:
            # We use place labels instead of compressed descriptors in order to enhance the connection
            # between compressed representation and gpm focus on retrieving highly informative mini-batches
            descriptors = descriptors.detach()
            compressed_descriptors = self.phead(descriptors)

            proxy_loss = self.phead.loss_fn(compressed_descriptors, labels)
            self.phead.optimizer.zero_grad()
            proxy_loss.backward(retain_graph=True)
            self.phead.optimizer.step()
            compressed_descriptors = compressed_descriptors.detach().cpu()
            self.pbank.update_bank(compressed_descriptors, labels)
            ids = self.pbank.build_index()
            self.trainer.train_dataloader = \
                DataLoader(dataset=self.trainer.train_dataloader.dataset,
                           batch_sampler=HardSampler(indexes_list=ids),
                           num_workers=args.num_workers)

        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}

    # For validation and test, we iterate step by step over the validation set
    def inference_step(self, batch):
        images, _ = batch
        descriptors = self(images)
        return descriptors.cpu().numpy().astype(np.float32)

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def validation_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.val_dataset)

    def test_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.test_dataset, self.num_preds_to_save)

    def inference_epoch_end(self, all_descriptors, inference_dataset, num_preds_to_save=0):
        """all_descriptors contains database then queries descriptors"""
        all_descriptors = np.concatenate(all_descriptors)
        queries_descriptors = all_descriptors[inference_dataset.database_num:]
        database_descriptors = all_descriptors[: inference_dataset.database_num]

        recalls, recalls_str = compute_recalls(
            inference_dataset, queries_descriptors, database_descriptors, num_queries_to_save=args.num_queries_to_save,
            output_folder=args.log_path, num_preds_to_save=num_preds_to_save,
            save_only_wrong_preds=self.save_only_wrong_preds, logger=trainer.logger
        )
        print(recalls_str)
        self.log('R@1', recalls[0], prog_bar=False, logger=True)
        self.log('R@5', recalls[1], prog_bar=False, logger=True)


def get_datasets_and_dataloaders(args):
    train_transform = tfm.Compose([
        # tfm.RandAugment(num_ops=3),
        tfm.GaussianBlur(10),   # High kernel size to make the blurring not too invalidating
        tfm.ColorJitter(brightness=0.7),
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = TrainDataset(
        dataset_folder=args.train_path,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        transform=train_transform
    )
    val_dataset = TestDataset(dataset_folder=args.val_path)
    test_dataset = TestDataset(dataset_folder=args.test_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


if __name__ == '__main__':
    args = parser.parse_arguments()
    kwargs = {"descriptors_dim": args.descriptors_dim, "num_preds_to_save": args.num_preds_to_save,
              "save_only_wrong_preds": args.save_only_wrong_preds}
    neptune_tags = []

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_datasets_and_dataloaders(args)
    if args.gpm:
        proxy_head = ProxyHead(out_dim=128, in_dim=args.descriptors_dim)
        proxy_bank = ProxyBank(k=4)
        kwargs.update({"proxy_bank": proxy_bank, "proxy_head": proxy_head})
        neptune_tags.append("gpm")

    if args.template_injection:
        neptune_tags.append("template injection")

    if args.reweighting:
        neptune_tags.append("contextual reweighting")

    if args.feature_mixing:
        mix = MixVPR(in_channels=128, in_h=28, in_w=28,
                     out_channels=2, out_rows=64, mix_depth=4)
        kwargs.update({"mix": mix})
        neptune_tags.append("mixvpr")
        if args.gpm:
            proxy_head = ProxyHead(out_dim=64, in_dim=128)
            proxy_bank = ProxyBank(k=4, dim=64)
            kwargs["proxy_head"] = proxy_head
            kwargs["proxy_bank"] = proxy_bank

    kwargs.update({"val_dataset": val_dataset, "test_dataset": test_dataset})
    if args.load_checkpoint:
        model = GeoModel.load_from_checkpoint(args.checkpoint_path + "/" + os.listdir(args.checkpoint_path)[-1])
    else:
        model = GeoModel(**kwargs)

    checkpoint_cb = ModelCheckpoint(
        monitor='R@1',
        filename='_epoch({epoch:02d})_step({step:04d})_R@1[{val/R@1:.4f}]_R@5[{val/R@5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        mode='max'
    )

    if args.neptune_api_key:
        neptune_logger = NeptuneLogger(
            api_key=args.neptune_api_key,  # replace with your own
            project="MLDL/geolocalization",  # format "workspace-name/project-name"
            tags=neptune_tags,
            log_model_checkpoints=True
        )
        PARAMS = {
            "batch_size": args.batch_size,
            "lr": 0.001,
            "max_epochs": args.max_epochs,
        }

        neptune_logger.log_hyperparams(params=PARAMS)
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[0],
            default_root_dir=args.log_path,  # Tensorflow can be used to viz
            num_sanity_val_steps=0,  # runs a validation step before stating training
            precision=16,  # we use half precision to reduce  memory usage
            max_epochs=args.max_epochs,
            check_val_every_n_epoch=1,  # run validation every epoch
            callbacks=[checkpoint_cb],
            reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
            log_every_n_steps=20,
            logger=neptune_logger
        )
    else:
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[0],
            default_root_dir=args.log_path,  # Tensorflow can be used to viz
            num_sanity_val_steps=0,  # runs a validation step before stating training
            precision=16,  # we use half precision to reduce  memory usage
            max_epochs=args.max_epochs,
            check_val_every_n_epoch=1,  # run validation every epoch
            callbacks=[checkpoint_cb],
            # we only run the checkpointing callback (you can add more)
            reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
            log_every_n_steps=20
        )

    # trainer.validate(model=model, dataloaders=val_loader)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader)
