import argparse
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data_interface import DInterface
from model import MInterface
from swanlab.integration.pytorch_lightning import SwanLabLogger
import swanlab


def train(args):
    if args['k_fold'] > 0:
        _cross_validation_train(args)
    else:
        _single_train(args)


def _single_train(args):
    swanlab.init(
        project="CT_MotionArtifact",
        config=args
    )
    swanlab_logger = SwanLabLogger(project="CT_MotionArtifact")

    # DataModule
    data_module = DInterface(
        csv_file=args['csv_file'],
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        aug_type=args.get('augment', 'none')
    )
    model = MInterface(**args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode='max',
        save_top_k=1,
        verbose=True
    )
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=20,
        mode="max",
        verbose=True
    )

    trainer = Trainer(
        max_epochs=args['epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=swanlab_logger,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
    )

    trainer.fit(model, datamodule=data_module)

    # print("训练完成，自动评估测试集...")
    # model_path = checkpoint_callback.best_model_path
    # if model_path:
    #     best_model = MInterface.load_from_checkpoint(model_path)
    #     trainer.test(best_model, datamodule=data_module)
    # else:
    #     print("未保存最佳模型，使用当前模型测试...")
    #     trainer.test(model, datamodule=data_module)

    swanlab.finish()


def _cross_validation_train(args):
    fold_results = []
    for fold in range(args['k_fold']):
        args["current_fold"] = fold
        seed_everything(42)
        try:
            swanlab.finish()  # 避免残留实验
        except RuntimeError:
            pass
        swanlab.init(
            project="CT_MotionArtifact",
            config=args,
            group=f"{args['k_fold']}-fold-cv",
            name=f"fold-{fold+1}"
        )

        data_module = DInterface(
            csv_file=args['csv_file'],
            batch_size=args['batch_size'],
            num_workers=args['num_workers'],
            aug_type=args.get('augment', 'none')
        )
        model = MInterface(**args)

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/fold_{fold+1}",
            monitor="val_acc",
            mode="max",
            filename="best-{epoch}-{val_acc:.2f}"
        )

        swanlab_logger = SwanLabLogger(
            project="CT_MotionArtifact",
            group=f"CV-{args['k_fold']}fold",
            name=f"fold{fold+1}"
        )

        trainer = Trainer(
            max_epochs=args['epochs'],
            callbacks=[checkpoint_callback],
            logger=swanlab_logger,
            deterministic=True,
            devices=1,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu'
        )

        trainer.fit(model, datamodule=data_module)
        model_path = checkpoint_callback.best_model_path
        best_model = MInterface.load_from_checkpoint(model_path)
        test_result = trainer.test(best_model, datamodule=data_module)
        fold_results.append(test_result[0]["test_acc"])
        swanlab.finish()

    # 输出 CV 结果
    final_metrics = {
        "cv_mean_acc": np.mean(fold_results),
        "cv_std_acc": np.std(fold_results),
        "cv_details": fold_results
    }
    print(f"\n{args['k_fold']}-Fold CV Results:\n{final_metrics}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="./data/labels.csv", help="CSV file path (path,label)")

    # 图像设置
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--image_channels", type=int, default=1)

    # 分类
    parser.add_argument("--class_num", type=int, default=2)

    # 模型
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument("--backbone", type=str, default='resnet18')
    parser.add_argument("--model_name", type=str, default="ResNet")

    # 训练
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=100)

    # 训练模式
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--k_fold", type=int, default=0)

    # 数据增强
    parser.add_argument("--augment", type=str, default="none", choices=["none", "light", "strong"])

    args = parser.parse_args()

    if args.mode == "train":
        train(vars(args))
