from os import path as osp
import subprocess

params = [
    # (4, 32, 512, 65536, 65536, 14),
    # (3, 8, 512, 32768, 65536, 8),
    # (3, 2, 128, 32768, 65536, 0),
    # (2, 16, 256, 65536, 65536, 3),
    (2, 2, 128, 65536, 65536, 14),
    # (4, 32, 512, 32768, 65536, 10),
    # (2, 32, 256, 65536, 65536, 14),
    # (2, 128, 128, 65536, 65536, 4),
    # (4, 16, 128, 65536, 65536, 8),
    # (2, 64, 256, 65536, 65536, 12),
    # (2, 8, 512, 65536, 65536, 20),
    # (3, 64, 256, 65536, 65536, 8),
]

files = [
    # "R1.mtx",
    # "R2.mtx",
    # "R3.mtx",
    # "R4.mtx",
    "R5.mtx",
    # "R6.mtx",
    # "R7.mtx",
    # "R8.mtx",
    # "R9.mtx",
    # "R10.mtx",
    # "R11.mtx",
    # "R12.mtx",
]

dir="/home/xx/data/rmat/"
project_dir='/home/xx/code/CESpGEMM'
exec_dirs=["build_1GPU",'build_2GPU','build_4GPU']
device_configs=['0','0,1','0,1,2,3']
output_dir='result_multigpu'

def run_all_parameters(exec, devices:str):
    for file, par in zip(files, params):
        args = [
            exec,
            "-DEVICES", devices,
            "-A", osp.join(dir, file),
            "-O", "test.bin",
            "-GIVE", "1",
            "-NZHEAD", str(par[0]),
            "-ZLEN", str(par[1]),
            "-SLEN", str(par[2]),
            "-BA", str(par[3]),
            "-BB", str(par[4]),
            "-GRATIO", str(par[5]),
            "-DEBUG", "1",
        ]
        result_lines = []
        proc=subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, universal_newlines=True)
        print(' '.join(args))
        stdout, stderr = proc.communicate()
        # print(proc.returncode)
        # print(stdout)
        # print(stderr)
        for line in stdout.splitlines():
            if line.startswith("exe_time="):
                result_lines.append(float(line[9:]))
        
        print(devices, file, result_lines[0])



if __name__ == "__main__":
    for exec_dir, devices in zip(exec_dirs, device_configs):
        run_all_parameters(osp.join(project_dir, exec_dir, 'compute'), devices)
        # break
