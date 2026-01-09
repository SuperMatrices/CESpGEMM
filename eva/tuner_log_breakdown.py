import argparse
import os
import os.path as osp

items = ['Sorting Time', 'Flop Count Time', 'Running Time']

def get_breakdown_times(file: str):
    stat = {k:[] for k in items}
    with open(file, "r") as f:
        lines = f.readlines()
        for l in lines:
            for k in items:
                if l.startswith(k):
                    stat[k].append(float(l[len(k)+1:].strip()))
    
    total_times = {k: sum(stat[k]) for k in items}
    return total_times

if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='')
    args = parser.parse_args()
    print("file,sort_time,count_time,run_time,tot_time,sort_ratio,count_ratio")
    names = sorted(os.listdir(args.d))
    for file in names:
        # print(file, osp.isfile(osp.join(args.d, file)))
        res = get_breakdown_times(osp.join(args.d, file))
        sort_t, count_t, run_t = res[items[0]], res[items[1]], res[items[2]]
        tot_t = sort_t + count_t + run_t
        print(f"{file},{sort_t:.12f},{count_t:.12f},{run_t:.12f},{tot_t:.12f},{sort_t/tot_t:.12f},{count_t/tot_t:.12f}")
        # print(res)输出时间和比例 3＋2列

