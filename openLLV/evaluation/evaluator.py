import math
import warnings

import torch
from typing import Dict, Any, Optional, Union, List, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
from pathlib import Path

from .baseMetric import BaseMetric
from openLLV.data import EvaluateDataset


class Evaluator:
    """Image quality evaluator"""

    def __init__(self,
                 en_img_dir: str,
                 ref_img_dir: Optional[str] = None,
                 save_path: Union[str, Path] = None,
                 metrics: Union[str, List[str]] = None,
                 device: str = None,
                 batch_size: int = 1,
                 num_workers: int = 8,
                 **kwargs):
        """
        Initialize evaluator

        Args:
            metrics: List of metrics to compute, default ['PSNR', 'SSIM'] if None
            device: computation device
            save_path: path/to/save.json
            **kwargs: parameters passed to each metric
        """
        if metrics is None:
            metrics = ['PSNR', 'SSIM']
        elif isinstance(metrics, str):
            metrics = [metrics.upper()]
        elif isinstance(metrics, list):
            metrics = [m.upper() for m in metrics]
        else:
            raise TypeError(f"Invalid type for_teach metrics parameter, expected str or List[str], but got {type(metrics)}")

        self.device = torch.device(device if device else
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.metric_instances = {}
        self.metric_order = []  # maintain metric computation order

        for metric_name in metrics:
            try:
                metric_name_lower = metric_name.lower()
                metric_mapping = {name.lower(): name for name in BaseMetric.list_available_metrics()}

                if metric_name_lower in metric_mapping.keys():
                    actual_name = metric_mapping[metric_name_lower]
                    metric_instance = BaseMetric.create_metric(
                        actual_name,
                        device=self.device,
                        **kwargs
                    )
                    simple_name = actual_name.replace('Metric', '')
                    self.metric_instances[simple_name] = metric_instance
                    self.metric_order.append(simple_name)
                else:
                    warnings.warn(f"Metric {metric_name} does not exist and has been skipped")

            except Exception as e:
                warnings.warn(f"Failed to create metric {metric_name}: {e}")

        print(f"Evaluator initialized. \n   - Metrics will be computed in order: {self.metric_order}")

        self.results = self.eval(
            en_img_dir=en_img_dir,
            ref_img_dir=ref_img_dir,
            save_path=save_path,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    def eval(self,
                en_img_dir: str,
                ref_img_dir: Optional[str] = None,
                save_path: Union[str, Path] = None,
                batch_size: int = 1,
                num_workers: int = 0) -> Dict[str, Any]:
        """
        Directly evaluate images in a folder

        Args:
            en_img_dir: enhanced image folder path
            ref_img_dir: reference image folder path (optional)
            save_path: save path
            batch_size: batch size
            num_workers: number of dataloader workers
            show_progress: whether to display progress bar

        Returns:
            evaluation result dictionary
        """

        dataset = EvaluateDataset(
            en_img_dir=en_img_dir,
            ref_img_dir=ref_img_dir,
        )

        print("\nDataset information:")
        print(f"  - Enhanced images: {en_img_dir} ({len(dataset.en_files)} images)")
        if ref_img_dir:
            ref_files = getattr(dataset, "ref_files", [])
            print(f"  - Reference images: {ref_img_dir} ({len(ref_files)} images)")
            print(f"  - Successfully paired: {len(dataset.paired_files)} pairs")
        else:
            print("  - No-reference mode")

        results = self.evaluate_dataset(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.save_results(results, save_path=save_path)

        return results

    def evaluate_dataset(self,
                         dataset: EvaluateDataset,
                         batch_size: int = 1,
                         num_workers: int = 8,) -> Dict[str, Any]:
        """
        Sequentially evaluate the entire datasets

        Args:
            dataset: EvaluateDataset instance
            batch_size: batch size
            num_workers: number of dataloader workers
            show_progress: whether to display progress bar

        Returns:
            dictionary containing:
                - 'metrics': metric results
                - 'filenames': filename list
                - 'statistics': statistics dictionary
        """

        filenames = self._get_dataset_filenames(dataset)
        results = {
            'metrics': {},
            'filenames': filenames,
            'statistics': {}
        }

        print(f"\n{'=' * 40}")
        for idx, metric_name in enumerate(self.metric_order, 1):
            metric = self.metric_instances[metric_name]

            better = "↑" if metric.higher_is_better else "↓"
            print(f"\nComputing metric [{idx}/{len(self.metric_order)}]: {metric_name} {better}, "
                  f"{'requires reference images' if metric.requires_reference else 'no-reference metric'}")

            if metric.requires_reference and not dataset.ref_dict:
                print(f"Skipping {metric_name} - reference images required but not provided")
                values = {filename: float('nan') for filename in filenames}
                results['metrics'][metric_name] = values
                results['statistics'][metric_name] = self._compute_metric_statistics(values, better)
                continue

            values = self._compute_metric_for_dataset(
                dataset=dataset, metric_name=metric_name, batch_size=batch_size,
                num_workers=num_workers
            )

            results['metrics'][metric_name] = values

            stats = self._compute_metric_statistics(values, better)
            results['statistics'][metric_name] = stats

            print(f"\n {metric_name} {stats['better']} statistics:")
            print(f"{'':<5} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Valid Samples':<15}")
            print(
                f"{'':<5} "
                f"{stats['mean']:<10.2f} "
                f"{stats['std']:<10.2f} "
                f"{stats['min']:<10.2f} "
                f"{stats['max']:<10.2f} "
                f"{stats['valid_count']:>7}/{stats['total_count']:<7}"
            )

        self.print_final_summary(results=results)

        return results

    @staticmethod
    def _get_dataset_filenames(dataset: EvaluateDataset) -> List[str]:
        if hasattr(dataset, "paired_files"):
            return [Path(path).name for path in dataset.paired_files]
        return [str(index) for index in range(len(dataset))]

    @staticmethod
    def collate_fn(batch):
        en_imgs = []
        ref_imgs = []
        names = []

        for en, ref, name in batch:
            en_imgs.append(en)
            ref_imgs.append(ref)  # may be None
            names.append(name)
        enBatch = torch.stack(en_imgs, dim=0)
        all_ref_none = all(r is None for r in ref_imgs)
        if all_ref_none:
            refBatch = None
        else:
            refBatch = torch.stack(ref_imgs, dim=0)

        return enBatch, refBatch, names


    def _compute_metric_for_dataset(self,
                                    dataset: EvaluateDataset,
                                    metric_name: str,
                                    batch_size: int = 1,
                                    num_workers: int = 0,) -> Dict[str, float]:
        """
        Compute a single metric over the entire datasets

        Args:
            dataset: EvaluateDataset instance
            metric_name: metric name
            batch_size: batch size
            num_workers: number of dataloader workers
            show_progress: whether to display progress bar

        Returns:
            dictionary of metric values (key: filename)
        """
        metric = self.metric_instances[metric_name]

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda',
            collate_fn=self.collate_fn
        )

        values = {}
        total_images = len(dataset)

        pbar = tqdm(total=total_images, desc=f"Computing {metric_name}", unit="batch",
                        bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')

        for batch_idx, (en_batch, ref_batch, name_batch) in enumerate(dataloader):
            batch_size_current = en_batch.size(0)

            for i in range(batch_size_current):
                en_img = en_batch[i:i + 1]
                ref_img = ref_batch[i:i + 1] if ref_batch is not None else None
                filename = name_batch[i]

                if not isinstance(filename, str):
                    filename = str(filename)

                try:
                    value = metric.compute(en_img, ref_img)
                    values[filename] = float(value)
                except Exception as e:
                    warnings.warn(f"Error computing {metric_name} for_teach image {filename}: {e}")
                    values[filename] = float('nan')

                pbar.update(1)

                processed = (batch_idx * batch_size) + i + 1
                if processed % 10 == 0 or processed == total_images:
                    pbar.set_postfix({
                        'batch_size': f'{batch_size}',
                        'progress': f"{processed}/{total_images}"})

        pbar.close()

        return values

    @staticmethod
    def _compute_metric_statistics(values: Dict[str, float], better: str) -> Dict[str, float]:
        """
        Compute statistics for_teach a single metric

        Args:
            values: metric values dictionary

        Returns:
            statistics dictionary
        """
        valid_values = []
        for k, v in values.items():
            if isinstance(v, float) and math.isnan(v):
                continue
            valid_values.append(v)

        if not valid_values:
            return {
                'mean': float('nan'),
                'min': float('nan'),
                'max': float('nan'),
                'std': float('nan'),
                'valid_count': 0,
                'total_count': len(values),
                'better': better ,
            }

        mean_value = sum(valid_values) / len(valid_values)
        min_value = min(valid_values)
        max_value = max(valid_values)

        variance = sum((x - mean_value) ** 2 for x in valid_values) / len(valid_values)
        std_value = math.sqrt(variance)

        return {
            'mean': float(mean_value),
            'min': float(min_value),
            'max': float(max_value),
            'std': float(std_value),
            'valid_count': len(valid_values),
            'total_count': len(values),
            'better': better,
        }

    def print_final_summary(self, results: Dict[str, Any],) -> None:
        """Print final summary"""
        print("\n" + "=" * 70)
        print(f"{'':<5} Evaluation completed - Final summary")
        print("-" * 70)

        print(f"{'':<5} {'  metric':<15} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")

        for metric_name, stats in results['statistics'].items():
            print(
                f"{'':<5} "
                f"{stats['better']:>2} "
                f"{metric_name:<13}"
                f"{stats['mean']:>10.2f} "
                f"{stats['std']:>10.2f} "
                f"{stats['min']:>10.2f} "
                f"{stats['max']:>10.2f}"
            )

        print("=" * 70)


    def save_results(self, results: Dict[str, Any], save_path: str = None):
        """
        Save evaluation results to file

        Args:
            results: evaluation results dictionary
            save_path: save path
        """
        if save_path is None:
            save_path = './results/eval.json'

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'metadata': {
                'device': str(self.device),
                'metrics': list(self.metric_instances.keys()),
                'total_images': len(results['filenames'])
            },
            'filenames': results['filenames'],
            'values': results['metrics'],
            'statistics': results['statistics']
        }

        serializable_data = self._make_serializable(save_data)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        print(f"\n"
              f"Evaluation results saved to: {save_path}")

    def _make_serializable(self, data: Any) -> Any:
        """Ensure data is serializable"""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, (int, float, str, bool)) or data is None:
            return data
        elif isinstance(data, torch.device):
            return str(data)
        else:
            return str(data)

    @classmethod
    def list_available_metrics(cls) -> List[str]:
        """Get all available metrics"""
        return BaseMetric.list_available_metrics(simple_names=True)
