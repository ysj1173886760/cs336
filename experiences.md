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

优化了一下，这里我为了方便直接用了python的heapdict，有点不讲武德，有时间再手搓一个吧。这里heap的更新用的是懒标记，就是更新heap的时候是直接把新的数据插进去，带一个version，然后pop的时候把version不对的都扔掉。一些缓存淘汰之类的策略里应该也比较常用
以及是可能简单一点就是分个块，然后每次只更新有变动的块，维护一下块内的最值，这样可以少遍历一些。

io consumption 0.00877690315246582
pre tokenize 0.8549258708953857
initial calc 0.026500940322875977
train executed in 2.859879 seconds
print time statistics
pre_tokenize 0.8549208641052246
calc_candidate 0.0023012161254882812
calc_occur_dict 2.8174686431884766

然后是tiny story:
io consumption 2.124450922012329
pre tokenize 59.56451368331909
initial calc 0.1409592628479004
train executed in 15.113764 seconds
print time statistics
pre_tokenize 59.56451106071472
calc_candidate 0.0029633045196533203
calc_occur_dict 14.929845809936523
总用用了1分多钟，最耗时的就是这个pre tokenize了

![](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20251001170823.png)

然后打印了一些单词表，看起来后面基本都是一些具体的词了。

然后问题中说最大的我这里算出来是：
In [4]: max(vocab.items(), key=lambda item: len(item[1]))
Out[4]: (7160, b' accomplishment')
accomplishment

(cs336-basics) ➜  assignment1-basics git:(main) ✗ grep accomplishment data/TinyStoriesV2-GPT4-train.txt | wc -l
    1510

我不确定答案对不对，就先放在这里了。


![](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20251001201120.png)

encode/decode，写的比较简单。for了一遍所有的merge，一个一个遍历，简单做了一个内存的优化。