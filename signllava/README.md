# Data
### Current version

One input (train) file for each visual representation. 
```
mock_data/
    - annotation.{train,dev}.json
    - dino.{train,dev}.h5
    - mae.{train,dev}.h5
    - pose.{train,dev}.h5
    - sign2vec.{train,dev}.h5
```
`annotation.{train, dev}.json` (*`clip_id` is an integer starting from 0.*)
```
{
    ${video_id}: {
                    ${clip_id}: {
                                    "translation": ...,
                                    "paraphrases": [A, B, C]
                                }
                    ....
                }
    ${video_id}: ....
}
```
`dino.{train,dev}.h5`
```
{
    ${video_id}: {
                    ${clip_name}: numpy.array,
                    ${clip_name}: numpy.array, ...}
    ${video_id}:...
}
```

### Future version
Multiple input files for each visual representation.

```
mock_data_multi/
    - annotation.{train,dev}.json
    - dino
        - dino.{train,dev}.{0,1,2,...}.h5
        - metadata_dino.{train,dev}.json
    - mae
        - dino.{train,dev}.{0,1,2,...}.h5
        - metadata_dino.{train,dev}.json
    - sign2vec
        - dino.{train,dev}.{0,1,2,...}.h5
        - metadata_dino.{train,dev}.json
    - pose
        - dino.{train,dev}.{0,1,2,...}.h5
        - metadata_dino.{train,dev}.json
```

`annotation.{train,dev}.json` (*`${clip_name}` is a string.*)
```
{
    ${video_id}: {
                "clip_order": [${clip_name}, ..., ],
                ${clip_name}: {
                                "translation": ....,
                                "paraphrases": [A, B, C, ...]},
                ${clip_name}: ....,
                },
    ${video_id}: ...
}
```
`dino/dino.train.${shard}.h5`
```
{
    ${video_id}: {
                    ${clip_name}: numpy.array,
                    ${clip_name}: numpy.array, ...}
    ${video_id}:...
}
```
`dino/metadata_dino.train.json`
```
{
    ${video_id}: ${shard}, ${video_id}: ${shard}, ....
}
```
# Pretraining
1. Modify the configuration file: 
```
signllava/configs/${filename}.yaml
```
2. Run pretraining:
```
bash signllava/scripts/pretrain.sh
```