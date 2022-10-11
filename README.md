# Fair_TrafficLightControl_DeepRL

In this repository, you will find the code to our paper:
"A Deep Reinforcement Learning Approach for Fair Traffic Signal Control" ([IEEE link](https://ieeexplore.ieee.org/abstract/document/9564847), [arxiv link](https://arxiv.org/pdf/2107.10146.pdf)).

### Summary:

In recent years, traffic control methods based on deep reinforcement learning (DRL) have gained attention due to their ability to exploit real-time traffic data, which is often poorly used by the traditional hand-crafted methods. While most recent DRL-based methods have focused on maximizing the throughput or minimizing the average travel time of the vehicles, the fairness of the traffic signal controllers has often been neglected. This is particularly important as neglecting fairness can lead to situations where some vehicles experience extreme waiting times, or where the throughput of a particular traffic flow is highly impacted by the fluctuations of another conflicting flow at the intersection. In order to address these issues, we introduce two notions of fairness: delay-based and throughput-based fairness, which correspond to the two issues mentioned above. Furthermore, we propose two DRL-based traffic signal control methods for implementing these fairness notions, that can achieve a high throughput as well.

### Citation:
```
@inproceedings{raeis2021deep,
  title={A deep reinforcement learning approach for fair traffic signal control},
  author={Raeis, Majid and Leon-Garcia, Alberto},
  booktitle={2021 IEEE International Intelligent Transportation Systems Conference (ITSC)},
  pages={2512--2518},
  year={2021},
  organization={IEEE}
}
```
