import csv
import json
import os
from datetime import datetime

LOG_PATH = os.environ.get(
    'EXP_LOG_PATH',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports', 'experiments_log.csv'),
)
FIELDS = ['exp_id', 'run_name', 'timestamp', 'description', 'params', 'dev_f1', 'test_f1', 'notes']


def log_experiment(exp_id: str, run_name: str, description: str, params: dict,
                   dev_f1: float, test_f1: float | None = None, notes: str = '') -> dict:
    """Ghi 1 dòng kết quả thí nghiệm vào reports/experiments_log.csv (tạo file nếu chưa có)."""
    row = {
        'exp_id': exp_id,
        'run_name': run_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'description': description,
        'params': json.dumps(params, ensure_ascii=False, default=str),
        'dev_f1': round(float(dev_f1), 4),
        'test_f1': '' if test_f1 is None else round(float(test_f1), 4),
        'notes': notes,
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)
    return row
