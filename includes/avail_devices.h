#pragma once
#include <vector>
#include <algorithm>
#include <cassert>

struct GPUSelection
{
    
    static std::vector<int>& get_device_map()
    {
        static std::vector<int> s;
        return s;
    }

    static void set_devices(const char* s)
    {
        std::vector<int>&avail_devices = get_device_map();
        int cur=0;
        for(int i=0;s[i]!=0;i++){
            const char c = s[i];
            if(c==','){
                avail_devices.push_back(cur);
                cur=0;
                continue;
            }
            if(isdigit(c)){
                cur = cur*10 + (c-'0');
            }
        }
        avail_devices.push_back(cur);
        std::sort(avail_devices.begin(), avail_devices.end());
        int u = std::unique(avail_devices.begin(), avail_devices.end()) - avail_devices.begin();
        avail_devices.resize(u);
    }
};

