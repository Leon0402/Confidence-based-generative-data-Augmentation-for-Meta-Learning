import argparse
import subprocess
import pandas as pd
import psutil
import time
import datetime
import yaml
from pathlib import Path


class UsageChecker:        
    def get_free_gpu(self, gpus):
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free,utilization.memory,utilization.gpu"])
        gpu_stats = [gpu.split(",") for gpu in gpu_stats.decode().split("\n")]
        gpu_df = pd.DataFrame(gpu_stats)
        gpu_df.columns = gpu_df.iloc[0, :] 
        gpu_df = gpu_df.iloc[1:-1,:]
        
        for col in gpu_df.columns:
            gpu_df[col] = gpu_df[col].str.replace(r'\D+', '', regex=True).astype(int)
            
        gpu_df["id"] = range(0, len(gpu_df))
        gpu_df = gpu_df[gpu_df["id"].isin(gpus)]
        return gpu_df

    def get_cpu_usage(self):
        return psutil.cpu_percent()

    def get_average_usage(self, gpus, n_iters, sleep_between_checks):
        gpu_usages = pd.DataFrame()
        cpu_usages = []
        for _ in range(n_iters):
            time.sleep(sleep_between_checks)
            gpu_usages = pd.concat([gpu_usages, self.get_free_gpu(gpus)])
            cpu_usages.append(self.get_cpu_usage())
    
        gpu_usages_avg = gpu_usages.groupby("id").mean()
        cpu_usage_avg = pd.Series(cpu_usages).max() * psutil.cpu_count(logical=True)
        return gpu_usages_avg, cpu_usage_avg


class ProcessHandler:
    
    def start_process(self, command):
        print(f"Process will be started with command: \n {command} \n\n")
        proc = subprocess.Popen([command], shell=True)
        return proc.pid

    def check_process_status(self, pid):
        return True if psutil.pid_exists(pid) else False
    

class TaskHandler:
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.process_handler = ProcessHandler()
        self.usage_checker = UsageChecker()
        self.config = self.load_config()
        
    def load_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
          
    def load_csv(self):
        self.task_df = pd.read_csv(self.config["csv_path"])
        
    def update_csv(self):
        self.task_df.to_csv(self.config["csv_path"], index=False)
    
    def check_availability(self, task, gpu_usages_avg, cpu_usage_avg):
        self.free_cpu_pct = psutil.cpu_count(logical=True)*100 - cpu_usage_avg
        self.free_gpu_memory = gpu_usages_avg[" memory.free [MiB]"]
        fits_on_cpu = self.free_cpu_pct > task["expected_cpu_usage (in %)"]  
        if fits_on_cpu:
            remaining_gpu_space = self.free_gpu_memory - task["expected_gpu_usage (in MB)"]  
            possible_gpus = remaining_gpu_space[remaining_gpu_space > 0]
            
            running_processes = self.task_df[self.task_df["status"] == "in progress"]
            processes_per_gpu = running_processes['attached to gpu'].astype(float).astype(int).value_counts()
            
            processes_per_gpu_over_process_limit = [gpu for gpu, processes in processes_per_gpu.items() if processes >= self.config["max_tasks_per_gpu"].get(str(gpu), 1000)]
            
            possible_gpus_final = possible_gpus[~possible_gpus.index.isin(processes_per_gpu_over_process_limit)]
            # check if there is a gpu that fits the task:
            if possible_gpus_final.shape[0] > 0:
                assigned_gpu_id = possible_gpus_final.index[possible_gpus_final.argmax()] #extract id of gpu that has the most space but can fit the script
                return assigned_gpu_id
            
        return -1
    
    def update_process_entry(self, new_df: pd.DataFrame):
        df = pd.read_csv(self.config["csv_path"], index_col="ID")
        df.update(new_df.set_index("ID"))
        df.to_csv(self.config["csv_path"])
    
    def assign_waiting_processes(self, gpu_usages_avg, cpu_usage_avg):        
        waiting_tasks = self.task_df[self.task_df["status"].str.strip() == "waiting"]
        waiting_tasks["assigned_gpus"] =  [self.check_availability(task, gpu_usages_avg, cpu_usage_avg) for idx, task in waiting_tasks.iterrows()]
        
        # filter for tasks that can fit on a GPU and the CPU
        waiting_tasks_gpu = waiting_tasks[waiting_tasks["assigned_gpus"] > -1]
        if waiting_tasks_gpu.shape[0] > 0:
            process_to_be_started = waiting_tasks_gpu.sort_values("expected_gpu_usage (in MB)", ascending=False).iloc[0]        
            cuda_str = f"CUDA_VISIBLE_DEVICES={process_to_be_started['assigned_gpus']} "
            
            # start process and update csv:
            process_in_task_df = self.task_df[self.task_df["ID"] == process_to_be_started["ID"]]
            process_in_task_df["attached to process"]  = self.process_handler.start_process(cuda_str+process_to_be_started["command"])
            process_in_task_df["attached to gpu"]  = process_to_be_started['assigned_gpus']
            process_in_task_df["status"]  = "in progress"
            process_in_task_df["timestamp_started"]  = datetime.datetime.now()
            process_in_task_df["last_timestamp"]  = datetime.datetime.now()
            process_in_task_df["process alive"]  = "True"
            print(f"Scheduled {process_to_be_started['ID']}")
            
            self.update_process_entry(process_in_task_df)


    def check_all_running_processes(self):
        running_processes = self.task_df[self.task_df["status"] == "in progress"]
        for idx, running_process in running_processes.iterrows():
            process_still_running = self.process_handler.check_process_status(int(float(running_process["attached to process"])))
            if not process_still_running:
                process_in_task_df = self.task_df[self.task_df["ID"] == running_process["ID"]]
                process_in_task_df["status"] = "exited"
                process_in_task_df["process alive"] = "False"
                process_in_task_df["last_timestamp"]  = datetime.datetime.now()
                print(f"Finshed {running_process['ID']}")
                
                self.update_process_entry(process_in_task_df)

    def orchestrate(self):
        while True:
            gpus = [int(gpu) for gpu, count in self.config["max_tasks_per_gpu"].items() if count > 0]
            gpu_usages_avg, cpu_usage_avg = self.usage_checker.get_average_usage(gpus, self.config["n_iters"],  self.config["sleep_between_checks"])
            
            self.config = self.load_config()
            self.load_csv()
            
            self.assign_waiting_processes(gpu_usages_avg, cpu_usage_avg)
            self.check_all_running_processes()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config_path', type=str, help='Path to config')
    args = parser.parse_args()
    
    th = TaskHandler(args.config_path)
    th.orchestrate()
    
# python cdmetadl/scheduler/TaskHandler.py --config_path "cdmetadl/scheduler/lab22/config.yaml"