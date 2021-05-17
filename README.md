# regional-analysis-paper
This is a repository for the paper "Learning from Precipitation Events in the Wider Domain to Improve the Performance of a Deep Learning-based Precipitation Nowcasting Model"

## Training and evaluation

### Trajectory GRU models

Learning with local data

```
cd run
./run_20210426_trajGRU_size200_wmse_alljapan.bash
```

Learning with data from all over the Japanese area

```
cd run

# Training
./run_20201025_trajGRU_size200_wmse_alljapan_1111.bash

# Testing
./run_20210116_trajGRU_size200_wmse_alljapan_test_fulldata.bash
```

Transfer leanining from All-Japan model

```
cd run
./run_20210426_trajGRU_size200_wmse_alljapan_trans.bash
```

### Other models

Evaluate with the persistence model.

```
cd run
./run_20210211_persistence_fulldata.bash
```

Evaluate with the Rainymotion semi-Lagrangian model.

```
cd run
./run_20210207_rainymotion_fulldata.bash
```

Evaluate with the JMA High Resolution Nowcast.

```
cd run
./run_20210502_hrncst_fulldata.bash
```
