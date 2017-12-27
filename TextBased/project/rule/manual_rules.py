# -*- coding: utf-8 -*-


import codecs 

def readText(filename, path=''):
    # 获取文本输入
    texts = []
    with codecs.open(filename, encoding='utf-8',mode='rU') as f:
        for line in f:
            texts.append(line.strip())
    return texts

def splitRules(rules,freq=5):
    l_r = []
    l_w = []
    for rule in rules:
        r = rule.split(',')
        if int(r[2]) >= freq:
            l_r.append(r[0])
            l_w.append(r[1])
    return l_r, l_w

def combination(s,i):
    elements = s.split(',')
    elements[2] = elements[2] + themes[i] + ';'
    elements[3] = elements[3] + words[i] + ';'
    elements[4] = elements[4] + senti[i] + ';'
    return ','.join(elements)

def matchRule(sentence):
    for idx,r in enumerate(rules):
        if sentence.find(r) > -1:
            return True,combination(sentence,idx)
            
    return False,sentence    


#filename = 'res1203_ver6_5weight_continueTrainedOn20000_2.csv'
filename = 'res1209_toupiao_mostnum_67314.csv'               ## 只需修改这里
rulename = 'rules.csv'

texts = readText(filename)
rules = readText(rulename)
rules, words = splitRules(rules)

rules.extend(['没什么效果','没有想象的好','没想象中的好'])
words.extend(['没效果','没有好','没好'])
themes = ['NULL'] * len(rules)
senti = ['-1'] * len(rules)
    
# 测试规则匹配的文本
s1 = '10118,东西收到了，快递慢，东西没想象中的好，声音小，音质也不好，不建议买。,快递;声音;音质;NULL;,慢;小;不好;不建议买;,-1;-1;-1;-1;'
s2 = '13776,上网速度比你原来的2G的还慢，一般般，没有想象的好，用用再说了,NULL;NULL;,慢;一般般;,-1;0;'
s3 = '13931,没有想象的好,,,'
#print(combination(s3,1))

# 只输出规则匹配成功的文本进行检查
with codecs.open('check.csv',encoding='utf-8',mode='w') as f:
    for result in map(matchRule, texts):
        if result[0] :
            print(result[1])
            f.write(result[1] + '\n')

# 输出全量的文本
with codecs.open('out1210v3.csv',encoding='utf-8',mode='w') as f:
    for result in map(matchRule, texts):
        try:
            f.write(result[1] + '\n')
        except:
            print('error:',result[1])
