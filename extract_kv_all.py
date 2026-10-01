import pandas as pd
import glob
import os

dirs = [d for d in glob.glob('results/*') if os.path.isdir(d)]
latest_dir = sorted(dirs, key=os.path.getmtime)[-1]
req_file = f'{latest_dir}/raw/requests.csv'
df_req = pd.read_csv(req_file)
df_req['timestamp'] = pd.to_datetime(df_req['timestamp'])

models = df_req['actual_quant_type'].unique()
results = []

for model in models:
    # 1. Đo cho CCU1 (S1, S2, S3)
    gpu_files_ccu1 = glob.glob(f'{latest_dir}/gpu/gpu_{model}_ccu1_*.csv')
    if gpu_files_ccu1:
        df_gpu1 = pd.read_csv(gpu_files_ccu1[0])
        df_gpu1['timestamp'] = pd.to_datetime(df_gpu1['timestamp'])
        idle_vram1 = df_gpu1['memory_used_mb'].min()
        
        reqs1 = df_req[(df_req['actual_quant_type'] == model) & (df_req['ccu'] == 1)]
        for s in ['S1', 'S2', 'S3']:
            s_reqs = reqs1[reqs1['scenario'] == s]
            if not s_reqs.empty:
                start_t = s_reqs['timestamp'].min() - pd.Timedelta(seconds=10)
                end_t = s_reqs['timestamp'].max() + pd.Timedelta(seconds=30)
                mask = (df_gpu1['timestamp'] >= start_t) & (df_gpu1['timestamp'] <= end_t)
                s_gpu = df_gpu1[mask]
                if not s_gpu.empty:
                    peak_vram = s_gpu['memory_used_mb'].max()
                    kv_cache = peak_vram - idle_vram1
                    results.append({'Model': model, 'Scenario': s, 'CCU': 1, 'KV_Cache_MB': kv_cache})

    # 2. Đo cho CCU2 (S4)
    gpu_files_ccu2 = glob.glob(f'{latest_dir}/gpu/gpu_{model}_ccu2_*.csv')
    if gpu_files_ccu2:
        df_gpu2 = pd.read_csv(gpu_files_ccu2[0])
        df_gpu2['timestamp'] = pd.to_datetime(df_gpu2['timestamp'])
        idle_vram2 = df_gpu2['memory_used_mb'].min()
        
        reqs2 = df_req[(df_req['actual_quant_type'] == model) & (df_req['ccu'] == 2)]
        for s in ['S4']:
            s_reqs = reqs2[reqs2['scenario'] == s]
            if not s_reqs.empty:
                start_t = s_reqs['timestamp'].min() - pd.Timedelta(seconds=10)
                end_t = s_reqs['timestamp'].max() + pd.Timedelta(seconds=30)
                mask = (df_gpu2['timestamp'] >= start_t) & (df_gpu2['timestamp'] <= end_t)
                s_gpu = df_gpu2[mask]
                if not s_gpu.empty:
                    peak_vram = s_gpu['memory_used_mb'].max()
                    kv_cache = peak_vram - idle_vram2
                    results.append({'Model': model, 'Scenario': s, 'CCU': 2, 'KV_Cache_MB': kv_cache})

df_res = pd.DataFrame(results)
pivot = df_res.pivot_table(index='Model', columns='Scenario', values='KV_Cache_MB').reset_index()
print(pivot.to_string(index=False))
