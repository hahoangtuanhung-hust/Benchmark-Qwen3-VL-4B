import pandas as pd
import glob
import os

dirs = [d for d in glob.glob('results/*') if os.path.isdir(d)]
latest_dir = sorted(dirs, key=os.path.getmtime)[-1]
req_file = f'{latest_dir}/raw/requests.csv'
df_req = pd.read_csv(req_file)

df_req['timestamp'] = pd.to_datetime(df_req['timestamp'])

# Lọc CCU1 và mô hình Q8_0
df_q8 = df_req[(df_req['actual_quant_type'] == 'Q8_0') & (df_req['ccu'] == 1)]

scenarios = df_q8['scenario'].unique()

gpu_file = glob.glob(f'{latest_dir}/gpu/gpu_Q8_0_ccu1_*.csv')[0]
df_gpu = pd.read_csv(gpu_file)
df_gpu['timestamp'] = pd.to_datetime(df_gpu['timestamp'])

idle_vram = df_gpu['memory_used_mb'].min()

for s in scenarios:
    s_reqs = df_q8[df_q8['scenario'] == s]
    start_time = s_reqs['timestamp'].min() - pd.Timedelta(seconds=10)
    end_time = s_reqs['timestamp'].max() + pd.Timedelta(seconds=30)
    
    # Filter GPU logs in this window
    mask = (df_gpu['timestamp'] >= start_time) & (df_gpu['timestamp'] <= end_time)
    s_gpu = df_gpu[mask]
    if not s_gpu.empty:
        peak_vram = s_gpu['memory_used_mb'].max()
        kv_cache = peak_vram - idle_vram
        print(f'Scenario {s}: Peak VRAM = {peak_vram} MB, Idle = {idle_vram} MB -> KV Cache = {kv_cache} MB')
