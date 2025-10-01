# Tokenizer

## TrainBPE

tests/test_train_bpe.py::test_train_bpe_speed 1
pre tokenize 0.022261857986450195
merges with 243 times, 1.4453072547912598

最简单的实现用了1.4秒已经可以过测试了。也可能是因为我并没有每次都重新做tokenize，直接复用了上一次的freq dict。（其实感觉这样写更简单，否则还得走一次encode的流程）

打了一下统计信息：
tests/test_train_bpe.py::test_train_bpe_speed 1
pre tokenize 0.022825956344604492
merges with 243 times, 1.4578559398651123
calc_freq 0.6579418182373047
calc_candidate 0.026823759078979492
calc_occur_dict 0.7456707954406738

因为计算pair/total的freq涉及到遍历全部数据，所以比较慢。这块assignment里已经给了提示，可以直接做一下增量的计算，因为每次只merge一个，大部分的频率是不需要重新计算的。

搞了一个增量的优化，写的其实也比较糊：
tests/test_train_bpe.py::test_train_bpe_speed 1
pre_tokenize 0.030631065368652344
calc_candidate 0.031034231185913086
calc_occur_dict 0.1312110424041748

这块其实还可以再降低一下常数，因为现在我是把每一个变化的word都重新算了一遍，其实只算增量的那点token就行。

然后有关special token的处理，记得要先split by special token，再去做pre tokenize

![](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20251001123110.png)

然后跑大数据集，做了一个tokenize的并行：
io consumption 0.009432077407836914
pre tokenize 0.973336935043335
initial calc 0.024617910385131836
pre_tokenize 0.9733319282531738
calc_candidate 6.40349555015564
calc_occur_dict 0.46723246574401855
发现现在瓶颈在calc_candidate上。用堆优化一下