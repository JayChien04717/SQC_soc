"""
Composite experiments — BatchExperiment and ParallelExperiment.

Inspired by IBM Qiskit Experiments ``BatchExperiment`` / ``ParallelExperiment``.

Usage
-----
::

    from QickworkspaceV2.core.composite import BatchExperiment

    # Declarative calibration pipeline — each step runs sequentially,
    # passing updated config forward.
    cal = BatchExperiment([
        ResonatorSpec(cfg),
        QubitSpec(cfg),
        PowerRabi(cfg),
    ])
    results = cal.run(py_avg=10)
    print(results["ResonatorSpec"].quality)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .experiment_data import ExperimentData, QualityFlag


class BatchExperiment:
    """
    Run a list of experiments sequentially, collecting all results.

    Each experiment is run in declaration order.  Results are stored in
    ``self.results`` (dict keyed by experiment type) and also returned
    from :meth:`run`.

    Parameters
    ----------
    experiments : list of BaseExperiment
        Experiment instances to run in order.
    stop_on_bad : bool, optional
        When ``True``, abort the batch if any experiment returns
        ``QualityFlag.BAD``.  Default is ``False``.
    """

    def __init__(
        self,
        experiments: List,
        stop_on_bad: bool = False,
    ):
        # Accept both bare experiments and (name, expt) tuples
        self._named: List = []
        for item in experiments:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                self._named.append(item)
            else:
                self._named.append((None, item))
        self.experiments = [expt for _, expt in self._named]
        self.stop_on_bad = stop_on_bad
        self.results: Dict[str, ExperimentData] = {}
        self._parent_id: Optional[str] = None

    def run(self, py_avg: int, **kwargs) -> Dict[str, ExperimentData]:
        """
        Run all experiments in order.

        Parameters
        ----------
        py_avg : int
            Software averages forwarded to every experiment.
        **kwargs
            Additional keyword arguments forwarded to each ``run()`` call.

        Returns
        -------
        results : dict
            Mapping ``experiment_type → ExperimentData``.
        """
        import time
        import uuid

        batch_id = str(uuid.uuid4())[:8]
        self.results = {}

        for i, (label, expt) in enumerate(self._named):
            expt_name = label or expt.EXPT_NAME or expt.__class__.__name__
            print(f"\n{'=' * 60}")
            print(f"  BatchExperiment [{i + 1}/{len(self.experiments)}] — {expt_name}")
            print(f"{'=' * 60}")

            result = expt.run(py_avg=py_avg, **kwargs)
            result.parent_id = batch_id
            self.results[expt_name] = result

            if self.stop_on_bad and result.quality == QualityFlag.BAD:
                print(
                    f"  [BatchExperiment] Stopping: {expt_name} returned BAD quality "
                    f"({result.quality_message})"
                )
                break
            time.sleep(0.5)
        return self.results

    def summary(self):
        """Print a quality summary table for all completed experiments."""
        print(f"\n{'=' * 60}")
        print("  BatchExperiment Summary")
        print(f"{'=' * 60}")
        for name, result in self.results.items():
            flag = result.quality.value.upper()
            msg = f" — {result.quality_message}" if result.quality_message else ""
            print(f"  {name:<40s} [{flag}]{msg}")
        print(f"{'=' * 60}")


class ParallelExperiment:
    """
    Run experiments in parallel threads and collect results.

    Useful when multiple independent experiments (different qubits, different
    parameter sweeps) can run simultaneously without hardware conflicts.

    Parameters
    ----------
    experiments : list of BaseExperiment
        Experiments to run in parallel.
    max_workers : int, optional
        Thread pool size.  Defaults to the number of experiments.
    """

    def __init__(
        self,
        experiments: List,
        max_workers: Optional[int] = None,
    ):
        # Accept both bare experiments and (name, expt) tuples
        self._named: List = []
        for item in experiments:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                self._named.append(item)
            else:
                self._named.append((None, item))
        self.experiments = [expt for _, expt in self._named]
        self.max_workers = max_workers or len(self.experiments)
        self.results: Dict[str, ExperimentData] = {}

    def run(self, py_avg: int, **kwargs) -> Dict[str, ExperimentData]:
        """Run all experiments concurrently, return collected results."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for label, expt in self._named:
                name = label or expt.EXPT_NAME or expt.__class__.__name__
                future = pool.submit(expt.run, py_avg, **kwargs)
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    self.results[name] = future.result()
                except Exception as exc:
                    print(f"  [ParallelExperiment] {name} raised: {exc}")
                    self.results[name] = ExperimentData(
                        experiment_type=name,
                        quality=QualityFlag.BAD,
                        quality_message=str(exc),
                    )

        return self.results
