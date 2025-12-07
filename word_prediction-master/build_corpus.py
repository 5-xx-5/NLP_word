#/usr/bin/python2
# coding: utf-8
'''
Builds an English news corpus from wikinews.
Feel free to use a prebuilt one such as reuter in nltk if you want,
but it may be too small.

Make sure you download raw wikinews data from 20160701 through 20170101 
at the following links before running this file,
then extract them to `data/raw` folder.

https://dumps.wikimedia.org/enwikinews/20160701/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20160720/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20160801/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20160820/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20160901/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20160920/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20161001/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20161020/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20161101/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20161120/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20161201/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20161220/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20170101/enwikinews-20170120-pages-articles-multistream.xml.bz2
'''
# from __future__ import print_function
# import numpy as np
# import pickle
# import codecs
# import lxml.etree as ET
# import regex
# from nltk.tokenize import sent_tokenize
#
# # def clean_text(text):
# #     text = regex.sub("\[http[^]]+? ([^]]+)]", r"\1", text)
# #     text = regex.sub("\[http[^]]+]", "", text)
# #     text = regex.sub("(?s)<ref>.+?</ref>", "", text) # remove reference links
# #     text = regex.sub("(?s)<[^>]+>", "", text) # remove html tags
# #     text = regex.sub("&[a-z]+;", "", text) # remove html entities
# #     text = regex.sub("(?s){{.+?}}", "", text) # remove markup tags
# #     text = regex.sub("(?s){.+?}", "", text) # remove markup tags
# #     text = regex.sub("(?s)\[\[([^]]+\|)", "", text) # remove link target strings
# #     text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text) # remove media links
# #
# #     text = regex.sub("[']{5}", "", text) # remove italic+bold symbols
# #     text = regex.sub("[']{3}", "", text) # remove bold symbols
# #     text = regex.sub("[']{2}", "", text) # remove italic symbols
# #
# #     text = regex.sub(u"[^ \r\n\p{Latin}\d\-'.?!]", " ", text)
# #     text = text.lower()
# #
# #     text = regex.sub("[ ]{2,}", " ", text) # Squeeze spaces.
# #     return text
#
# def clean_text(text):
#     text = regex.sub(r"\[http[^]]+? ([^]]+)]", r"\1", text)
#     text = regex.sub(r"\[http[^]]+]", "", text)
#     text = regex.sub(r"(?s)<ref>.+?</ref>", "", text)
#     text = regex.sub(r"(?s)<[^>]+>", "", text)
#     text = regex.sub(r"&[a-z]+;", "", text)
#     text = regex.sub(r"(?s){{.+?}}", "", text)
#     text = regex.sub(r"(?s){.+?}", "", text)
#     text = regex.sub(r"(?s)\[\[([^]]+\|)", "", text)
#     text = regex.sub(r"(?s)\[\[([^]]+:.+?]])", "", text)
#
#     text = regex.sub(r"[']{5}", "", text)
#     text = regex.sub(r"[']{3}", "", text)
#     text = regex.sub(r"[']{2}", "", text)
#
#     text = regex.sub(r"[^ \r\n\p{Latin}\d\-'.?!]", " ", text)
#     text = text.lower()
#     text = regex.sub(r"[ ]{2,}", " ", text)
#     return text
#
# def build_corpus():
#     import glob
#
#     with codecs.open('data/en_wikinews.txt', 'w', 'utf-8') as fout:
#         fs = glob.glob('data/raw/*.xml')
#         ns = "{http://www.mediawiki.org/xml/export-0.10/}" # namespace
#         print("找到的 XML 文件列表：", fs)
#         for f in fs:
#             i = 10
#             for _, elem in ET.iterparse(f, tag=ns+"text"):
#                 try:
#                     if i > 5:
#                         running_text = elem.text
#                         running_text = running_text.split("===")[0]
#                         running_text = clean_text(running_text)
#                         paras = running_text.split("\n")
#                         if running_text is None:
#                             print("Empty text")
#                             continue
#                         for para in paras:
#                             if len(para) > 500:
#                                 sents = [regex.sub("([.!?]+$)", r" \1", sent) for sent in sent_tokenize(para.strip())]
#                                 fout.write(" ".join(sents) + "\n")
#                 except:
#                     continue
#
#                 elem.clear() # We need to save memory!
#                 i += 1
#                 if i % 1000 == 0: print(i,)
#
# if __name__ == '__main__':
#     build_corpus()
#     print("Done")

# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import bz2
import codecs
import regex
import lxml.etree as ET
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab')



# ----------------------------------
# Step 1: 清理文本
# ----------------------------------
def clean_text(text):
    if not text:
        return ""

    # 去掉 ref、模板、标签、链接
    text = regex.sub(r"(?s)<ref.*?>.*?</ref>", " ", text)
    text = regex.sub(r"(?s)<[^>]+>", " ", text)
    text = regex.sub(r"(?s)\{\{.*?\}\}", " ", text)
    text = regex.sub(r"(?s)\{.*?\}", " ", text)
    text = regex.sub(r"\[\[File:[^\]]+]]", " ", text)
    text = regex.sub(r"\[\[[^\|\]]+\|([^\]]+)\]\]", r"\1", text)  # [[A|B]] → B
    text = regex.sub(r"\[\[[^\]]+]]", " ", text)

    # HTML 转义字符
    text = regex.sub(r"&[a-z]+;", " ", text)

    # 保留英文字符
    text = regex.sub(r"[^a-zA-Z0-9\.\,\!\?\:\;\-\(\) ]", " ", text)

    # 多空格变一
    text = regex.sub(r" +", " ", text)

    return text.strip()


# ----------------------------------
# Step 2: 自动解压 .bz2（如果需要）
# ----------------------------------
def ensure_xml(filepath):
    if filepath.endswith(".bz2"):
        xml_path = filepath[:-4]  # 去掉 .bz2
        print("解压中：", filepath)
        with bz2.open(filepath, 'rb') as f_in, open(xml_path, 'wb') as f_out:
            f_out.write(f_in.read())
        print("解压完成：", xml_path)
        return xml_path
    else:
        return filepath


# ----------------------------------
# Step 3: 解析 XML 的 <text> 标签
# ----------------------------------
def extract_text(xml_path, fout):
    ns = "{http://www.mediawiki.org/xml/export-0.11/}"

    print("开始解析 XML：", xml_path)

    count = 0
    for _, elem in ET.iterparse(xml_path, tag=ns + "text"):
        raw = elem.text
        elem.clear()  # 释放内存

        if not raw:
            continue

        # 清洗
        clean = clean_text(raw)
        if not clean:
            continue

        # 分句
        sents = sent_tokenize(clean)
        for s in sents:
            if len(s.strip()) > 20:   # 避免太短的语句
                fout.write(s.strip() + "\n")

        count += 1

        if count % 2000 == 0:
            # print("已解析 text 标签数量：", count)
            print("Number of text tags parsed: ", count)

    # print("完成解析：共提取文本段落", count)
    print("Parsing completed: total extracted text paragraphs", count)


# ----------------------------------
# Step 4: 主函数
# ----------------------------------
def build_corpus():
    input_dir = "data/inpt"
    output_file = "data/en_wikinews.txt"

    with codecs.open(output_file, "w", "utf-8") as fout:
        for name in os.listdir(input_dir):
            filepath = os.path.join(input_dir, name)

            if name.endswith(".xml") or name.endswith(".xml.bz2"):
                xml_path = ensure_xml(filepath)
                extract_text(xml_path, fout)

    print("All extraction completed! Output file: ", output_file)


if __name__ == "__main__":
    build_corpus()
