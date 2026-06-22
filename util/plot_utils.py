"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path, PurePath


# def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
#     '''
#     Function to plot specific fields from training log(s). Plots both training and test results.

#     :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
#               - fields = which results to plot from each log file - plots both training and test for each field.
#               - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
#               - log_name = optional, name of log file if different than default 'log.txt'.

#     :: Outputs - matplotlib plots of results in fields, color coded for each log file.
#                - solid lines are training results, dashed lines are test results.

#     '''
#     func_name = "plot_utils.py::plot_logs"

#     # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
#     # convert single Path to list to avoid 'not iterable' error

#     if not isinstance(logs, list):
#         if isinstance(logs, PurePath):
#             logs = [logs]
#             print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
#         else:
#             raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
#             Expect list[Path] or single Path obj, received {type(logs)}")

#     # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
#     for i, dir in enumerate(logs):
#         if not isinstance(dir, PurePath):
#             raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
#         if not dir.exists():
#             raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
#         # verify log_name exists
#         fn = Path(dir / log_name)
#         if not fn.exists():
#             print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
#             print(f"--> full path of missing log file: {fn}")
#             return

#     # load log file(s) and plot
#     dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

#     fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

#     for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
#         for j, field in enumerate(fields):
#             if field == 'mAP':
#                 coco_eval = pd.DataFrame(
#                     np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
#                 ).ewm(com=ewm_col).mean()
#                 axs[j].plot(coco_eval, c=color)
#             else:
#                 for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
#                     print("\n====== 新的 log 文件 ======")
#                     print(df.head())          # 打印前几行看看内容
#                     print(df.dtypes)          # 打印每列的数据类型
#                     print(df.columns)         # 打印列名
#                 df.interpolate().ewm(com=ewm_col).mean().plot(
#                     y=[f'train_{field}', f'test_{field}'],
#                     ax=axs[j],
#                     color=[color] * 2,
#                     style=['-', '--']
#                 )
#     for ax, field in zip(axs, fields):
#         ax.legend([Path(p).name for p in logs])
#         ax.set_title(field)
def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    """
    Plot training logs safely (handles non-numeric columns) and prints which columns are plotted/skipped.
    """
    func_name = "plot_utils.py::plot_logs"

    # 确保 logs 是 list
    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # 检查目录和文件
    for dir in logs:
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument: {dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}. Have you gotten to Epoch 1 in training?\n--> full path: {fn}")
            return

    # 读取 log 文件
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                # mAP 特殊处理
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
                print(f"字段 {field}: 使用 test_coco_eval_bbox 绘制 mAP 曲线")
            else:
                # 普通字段，只画数字列
                train_col = f"train_{field}"
                test_col = f"test_{field}"
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                cols_to_plot = [col for col in [train_col, test_col] if col in numeric_cols]

                skipped_cols = [col for col in [train_col, test_col] if col not in numeric_cols]

                if skipped_cols:
                    print(f"字段 {field}: 跳过非数字列 {skipped_cols}")
                if not cols_to_plot:
                    print(f"字段 {field}: 没有数字列可画，跳过")
                    continue

                print(f"字段 {field}: 绘制列 {cols_to_plot}")
                df[cols_to_plot].interpolate().ewm(com=ewm_col).mean().plot(
                    y=cols_to_plot,
                    ax=axs[j],
                    color=[color]*len(cols_to_plot),
                    style=['-', '--'][:len(cols_to_plot)]
                )

    # 设置标题和图例
    for ax, field in zip(axs, fields):
        ax.set_title(field)
        ax.legend([Path(p).name for p in logs])


# def plot_precision_recall(files, naming_scheme='iter'):
#     if naming_scheme == 'exp_id':
#         # name becomes exp_id
#         names = [f.parts[-3] for f in files]
#     elif naming_scheme == 'iter':
#         names = [f.stem for f in files]
#     else:
#         raise ValueError(f'not supported {naming_scheme}')
#     fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
#     for f in files:
#         data = torch.load(f, weights_only=False)
#         if 'precision' not in data or data['precision'].numel() == 0:
#             print(f"{f.name}: precision 数据为空，跳过绘制")
#             continue
#         precision = data['precision']
#         recall = data['params'].recThrs
#         print(f"{f.name}: precision shape={precision.shape}, recall shape={recall.shape}")
#     for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
#         data = torch.load(f, weights_only=False)
#         # precision is n_iou, n_points, n_cat, n_area, max_det
#         precision = data['precision']
#         recall = data['params'].recThrs
#         scores = data['scores']
#         # take precision for all classes, all areas and 100 detections
#         precision = precision[0, :, :, 0, -1].mean(1)
#         scores = scores[0, :, :, 0, -1].mean(1)
#         prec = precision.mean()
#         rec = data['recall'][0, :, 0, -1].mean()
#         print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
#               f'score={scores.mean():0.3f}, ' +
#               f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
#               )
#         axs[0].plot(recall, precision, c=color)
#         axs[1].plot(recall, scores, c=color)

#     axs[0].set_title('Precision / Recall')
#     axs[0].legend(names)
#     axs[1].set_title('Scores / Recall')
#     axs[1].legend(names)
#     return fig, axs

def plot_precision_recall(files, naming_scheme='iter'):
    """
    安全版 PR 曲线绘制，自动检查 precision/recall 数据，避免空图报错
    """
    if naming_scheme == 'exp_id':
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')

    if len(files) == 0:
        print("没有找到 eval 文件，PR 曲线无法绘制")
        return None, None

    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))

    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f, weights_only=False)

        # 检查 precision 是否存在并非空
        if 'precision' not in data:
            print(f"{f.name}: 没有 precision 数据，跳过")
            continue
        precision = data['precision']
        if isinstance(precision, np.ndarray):
            if precision.size == 0:
                print(f"{f.name}: precision 为空，跳过")
                continue
        elif hasattr(precision, 'numel'):  # torch.Tensor
            if precision.numel() == 0:
                print(f"{f.name}: precision 为空，跳过")
                continue
        else:
            print(f"{f.name}: precision 类型未知 {type(precision)}, 跳过")
            continue

        # 检查 recall
        recall = data['params'].recThrs if 'params' in data else None
        if recall is None or (isinstance(recall, np.ndarray) and recall.size == 0):
            print(f"{f.name}: recall 数据为空，跳过")
            continue

        # 处理 precision 维度
        try:
            precision_plot = precision[0, :, :, 0, -1].mean(1)  # 按原脚本
        except Exception as e:
            print(f"{f.name}: precision 索引失败 {e}, 跳过")
            continue

        scores_plot = data['scores'][0, :, :, 0, -1].mean(1)
        rec = data['recall'][0, :, 0, -1].mean()

        prec = precision_plot.mean()
        print(f'{naming_scheme} {name}: mAP@50={prec*100:05.1f}, score={scores_plot.mean():0.3f}, f1={2*prec*rec/(prec+rec+1e-8):0.3f}')

        # 绘图
        axs[0].plot(recall, precision_plot, c=color)
        axs[1].plot(recall, scores_plot, c=color)

    axs[0].set_title('Precision / Recall')
    axs[1].set_title('Scores / Recall')
    axs[0].legend(names)
    axs[1].legend(names)

    return fig, axs

if __name__ == '__main__':
    files = list(Path('/home/jiangyang.li2/detr_hemlet_check/Deformable-DETR/output_df/eval').glob('*.pth'))
    plot_precision_recall(files)
    plt.show()
    plt.savefig('/home/jiangyang.li2/detr_hemlet_check/Deformable-DETR/output_df/training_curves1.png', dpi=300)

    plot_logs(logs=Path('/home/jiangyang.li2/detr_hemlet_check/Deformable-DETR/output_df'),fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt')
    plt.show()
    plt.savefig('/home/jiangyang.li2/detr_hemlet_check/Deformable-DETR/output_df/training_curves2.png', dpi=300)
