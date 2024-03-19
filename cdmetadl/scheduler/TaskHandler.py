import argparse
import subprocess
import pandas as pd
import psutil
import time
import datetime

class UsageChecker:
    
    def __init__(self, use_gpus):
        self.use_gpus = use_gpus
        
    def get_free_gpu(self):
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free,utilization.memory,utilization.gpu"])
        gpu_stats = [gpu.split(",") for gpu in gpu_stats.decode().split("\n")]
        gpu_df = pd.DataFrame(gpu_stats)
        gpu_df.columns = gpu_df.iloc[0, :] 
        gpu_df = gpu_df.iloc[1:-1,:]
        
        for col in gpu_df.columns:
            gpu_df[col] = gpu_df[col].str.replace(r'\D+', '', regex=True).astype(int)
            
        gpu_df["id"] = range(0, len(gpu_df))
        gpu_df = gpu_df[gpu_df["id"].isin(self.use_gpus)]
        return gpu_df

    def get_cpu_usage(self):
        return psutil.cpu_percent()

    def get_average_usage(self, n_iters=20, sleep_between_checks=5):
        gpu_usages = pd.DataFrame()
        cpu_usages = []
        for _ in range(n_iters):
            time.sleep(sleep_between_checks)
            gpu_usages = pd.concat([gpu_usages, self.get_free_gpu()])
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
    
    def __init__(self, csv_path, use_gpus, sleep_between_checks=5):
        self.csv_path = csv_path
        self.task_df = pd.read_csv(self.csv_path)
        self.usage_checker = UsageChecker(use_gpus)
        self.sleep_between_checks = sleep_between_checks
        self.process_handler = ProcessHandler()
        
    def load_csv(self):
        self.task_df = pd.read_csv(self.csv_path)
        
    def update_csv(self):
        self.task_df.to_csv(self.csv_path, index=False)
    
    def check_availability(self, task, gpu_usages_avg, cpu_usage_avg):
        self.free_cpu_pct = psutil.cpu_count(logical=True)*100 - cpu_usage_avg
        self.free_gpu_memory = gpu_usages_avg[" memory.free [MiB]"]
        fits_on_cpu = self.free_cpu_pct > task["expected_cpu_usage (in %)"]  
        if fits_on_cpu:
            remaining_gpu_space = self.free_gpu_memory - task["expected_gpu_usage (in MB)"]  
            possible_gpus = remaining_gpu_space[remaining_gpu_space > 0]
            # check if there is a gpu that fits the task:
            if possible_gpus.shape[0] > 0:
                assigned_gpu_id = possible_gpus.argmax() #extract id of gpu that has the most space but can fit the script
                return assigned_gpu_id
            
        return -1
    
    def assign_waiting_processes(self):
        gpu_usages_avg, cpu_usage_avg = self.usage_checker.get_average_usage(sleep_between_checks=self.sleep_between_checks)
        waiting_tasks = self.task_df[self.task_df["status"].str.strip() == "waiting"]
        waiting_tasks["assigned_gpus"] =  [self.check_availability(task, gpu_usages_avg, cpu_usage_avg) for idx, task in waiting_tasks.iterrows()]

        # filter for tasks that can fit on a GPU and the CPU
        waiting_tasks_gpu = waiting_tasks[waiting_tasks["assigned_gpus"] > -1]
        
        if waiting_tasks_gpu.shape[0] > 0:
            # select the first task stat is waiting and can fit on any GPU and start it (can be replaced by a more advanced logic like best fit first etc.)
            process_to_be_started = waiting_tasks_gpu.iloc[0]
            cuda_str = f"CUDA_VISIBLE_DEVICES={process_to_be_started['assigned_gpus']} "
                
            # start process and update csv:
            process_in_task_df = self.task_df[self.task_df["ID"] == process_to_be_started["ID"]]
            process_in_task_df["attached to process"]  = self.process_handler.start_process(cuda_str+process_to_be_started["command"])
            process_in_task_df["attached to gpu"]  = process_to_be_started['assigned_gpus']
            process_in_task_df["status"]  = "in progress"
            process_in_task_df["timestamp_started"]  = datetime.datetime.now()
            process_in_task_df["last_timestamp"]  = datetime.datetime.now()
            process_in_task_df["process alive"]  = "True"
            self.task_df.update(process_in_task_df)


    def check_all_running_processes(self):
        running_processes = self.task_df[self.task_df["status"] == "in progress"]
        for idx, running_process in running_processes.iterrows():
            process_still_running = self.process_handler.check_process_status(int(float(running_process["attached to process"])))
            if not process_still_running:
                process_in_task_df = self.task_df[self.task_df["ID"] == running_process["ID"]]
                process_in_task_df["status"] = "exited"
                process_in_task_df["process alive"] = "False"
                process_in_task_df["last_timestamp"]  = datetime.datetime.now()
                self.task_df.update(process_in_task_df)


    def orchestrate(self):
        while True:
            self.load_csv()
            original_df = self.task_df.copy()
            self.assign_waiting_processes()
            self.check_all_running_processes()
            
            if (original_df != self.task_df).any().any():
                self.update_csv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--use_gpus', default=[0, 1, 2], type=list, help='List of GPU IDs to be used')
    parser.add_argument('--csv_path', default="./baselines.csv", type=str, help='Path to CSV file')
    args = parser.parse_args()
    
    th = TaskHandler(args.csv_path, args.use_gpus, sleep_between_checks=10)
    th.orchestrate()