# BPE Tokenizer

## Problem (unicode1): Understanding Unicode (1 point)

### (a) What Unicode character does chr(0) return?

![](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20250924104751.png)

0这个码点代表的字符

### (b) How does this character’s string representation (__repr__()) differ from its printed representation?

![](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20250924104941.png)

print用的是__str__，repr更精确，而__str__可读性更强，但是可能有歧义

![](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20250924105134.png)

### (c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

![](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20250924105252.png)

chr(0)代表的是空字符，所以print的时候是空的。

而REPL模式下，是通过repr打印的，方便debug。所以显示的是`\x00`

## Problem (unicode2): Unicode Encodings

### (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

![](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20250924125320.png)

utf8的编码可以生成更少的byte，节省计算资源。生成更多的byte更可能导致tokenize的时候生成更多的token，从而消耗更多计算资源。

### (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

因为这里是根据byte单独解码的，而utf8是变长编码，不一定每一个code point都对应一个字节。

![](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20250924125700.png)

### (c) Give a two byte sequence that does not decode to any Unicode character(s).

这里不确定说的是不是utf8的编码规则。简单操作就是for一遍，然后decode失败的时候就打印出来就行

![](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20250924130105.png)

