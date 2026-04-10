for metric in accuracy distance f1; do
  env/bin/python src/compute_$metric.py
done
