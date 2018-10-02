'''
南园
十三
李贺
男儿
何不
不带
带
吴钩
收取
关山
五十
凌烟阁
书生
万户侯
'''
# 自己填字典路径，字典内容见上面注释。
DICT_PATH=r"./dic-utf-8.mydic"

# 正向最大匹配.
# 找到字典里最长词汇的长度M，然后从文章的开头开始，每次切分出M长度的片段，
# 如果该片段在字典里，则接着看片段后面的内容，
# 如果该片段不再字典里，去掉片段最后一个字，再看这个新的片段是否在字典里，如果在，接着处理这个新片段后面的内容。
# 片段处理完，如果都没发现字典里存在相应片段，则从该片段开头位置后移一个位置，继续处理大小为M的片段，
# 直到文章处理完毕。
class MM(object):
    def __init__(self,dict_path):
        self.maximum=0
        self.dictionary=set()
        with open(dict_path,"r",encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line=="":
                    continue
                else:
                    self.dictionary.add(line)
                    if len(line)>self.maximum:
                        self.maximum=len(line)

    # 分词.
    # individual_character表示如果某个字不在字典里,是否将这个词存到结果里,默认存.
    def cut(self,text,individual_character=True):
        index=0
        end=len(text)
        result=[]
        while index<end:
            word=""
            for size in range(self.maximum,0,-1):
                if index+size>end:
                    continue
                piece=text[index:index+size]
                if piece in self.dictionary:
                    word=piece
                    result.append(word)
                    index+=size
                    break
            if word=="":
                if individual_character:
                    word=text[index]
                    result.append(word)
                index+=1

        return result

# 逆向最大匹配.
# 类似正向匹配，差别在于从文章末尾开始，取M大小的片段，
# 为找到相应片段，则将当前片段第一个字去掉，继续在字典里找。
# 将片段不断缩小，知道缩小到0，即处理完，仍然没有在字典中找到相应片段，
# 则将片段末尾的那个位置往前移动一位，继续处理包括这个末尾的前M个字。
class RMM(object):
    def __init__(self,dict_path):
        self.maximum=0 # 词典中词汇的最大长度.
        # 读取词典到self.dictionary里,并获得词典中词汇最大长度.
        self.dictionary=set()
        with open(dict_path,"r",encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line=="":
                    continue
                else:
                    self.dictionary.add(line)
                    if len(line)>self.maximum:
                        self.maximum=len(line)

    # 分词.
    # individual_character表示如果某个字不在字典里,是否将这个词存到结果里,默认存.
    def cut(self,text,individual_character=True):
        # 从后向前处理text.
        # index为当前处理到的字的位置的序号.要对index之前maximum个字进行处理.
        index=len(text)
        result=[]
        while index>0:
            word=""
            # size为当前处理的字串长度,范围为maximum到1.
            for size in range(self.maximum,0,-1):
                # 当前大小的字串不存在,则缩小size,即continue.
                if index-size<0:
                    continue
                # 当前大小的字串存在,则与字典中词汇进行比较.
                piece=text[index-size:index]#从text中切片出当前字串.
                # 如果当前字串存在与字典中,则存入结果,并更改index位置.
                if piece in self.dictionary:
                    word=piece
                    result.append(word)
                    index-=size
                    break
            # 如果各种size都尝试完毕,且word仍然等于"",则说明index没有减过size,所以此时index需要缩小1.
            if word=="":
                index-=1#因为只有index大于0,才会进入循环,所以index-1一定大于等于0.
                
                if individual_character:
                    #因为index减1之后,index位置在之后代码中不会被处理,
                    #又因为之前代码并没有给index所在位置匹配到相应词,所以这个单字应该被单独存储下来.
                    word=text[index]
                    result.append(word)

        #因为处理顺序时逆序的,所以后面的词会存在result的前面,所以返回时需要掉转顺序.
        return result[::-1]
    
def main():
    text="南园十三首其五李贺男儿何不带吴钩收取关山五十州请君暂上凌烟阁若个书生万户侯"

    # 正向最大匹配法.
    tokenizer=MM(DICT_PATH)
    print(tokenizer.cut(text,individual_character=True))
    print(tokenizer.cut(text,individual_character=False))

    # 逆向最大匹配法.
    tokenizer=RMM(DICT_PATH)
    print(tokenizer.cut(text,individual_character=True))
    print(tokenizer.cut(text,individual_character=False))


main()

'''
# 注意比较正向与逆向的区别，比如“何不”那一句。
# 正向
['南园', '十三', '首', '其', '五', '李贺', '男儿', '何不', '带', '吴钩', '收取', '关山', '五十', '州', '请', '君', '暂', '上', '凌烟阁', '若', '个', '书生', '万户侯']
['南园', '十三', '李贺', '男儿', '何不', '带', '吴钩', '收取', '关山', '五十', '凌烟阁', '书生', '万户侯']
# 逆向
['南园', '十三', '首', '其', '五', '李贺', '男儿', '何', '不带', '吴钩', '收取', '关山', '五十', '州', '请', '君', '暂', '上', '凌烟阁', '若', '个', '书生', '万户侯']
['南园', '十三', '李贺', '男儿', '不带', '吴钩', '收取', '关山', '五十', '凌烟阁', '书生', '万户侯']
'''
