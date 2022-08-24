from benchopt.benchmark import Benchmark
from benchopt import run_benchmark

benchmark_logreg = Benchmark('.')

run_benchmark(benchmark_logreg, max_runs=200,
              n_jobs=1, n_repetitions=1,
              solver_names=["sklearn", 'skglm', "skglm-gram"],
              dataset_names=['simulated', 'libsvm'])
